// Mush8 - interp.rs
//! NX_INTERP=1: frame interpolation. The game still simulates at 60 Hz; after each frame its commands are replayed a second
//! time with every changed uniform buffer (matrices, positions, bone palettes...) blended halfway between this frame and
//! the last, and that image is shown half a frame before the real one.
//! Draws are paired across frames by their mesh (program, vertex and index buffers, counts) and order, and among several
//! draws of the same mesh by whose uniforms are closest. The game double-buffers its uniforms, so the same draw reads a
//! different address each frame and the bytes are kept here when it draws.
use super::cmd::{Cmd, TexBuilder};
use super::state::GState;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// Replay diagnostics: blended buffers bound, and draws that ran / waited on a shader / failed.
pub static BOUND: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static OUTCOMES: [std::sync::atomic::AtomicUsize; 3] = [const { std::sync::atomic::AtomicUsize::new(0) }; 3];

/// NX_INTERP_COMPUTE=1 runs compute work again in the in-between replay. Off by default: compute here keeps state on the
/// GPU (exposure adaptation and the like), and running it twice a frame moves that state twice as fast.
pub static KEEP_COMPUTE: LazyLock<bool> = LazyLock::new(|| std::env::var("NX_INTERP_COMPUTE").is_ok_and(|v| v != "0"));

/// NX_INTERP_TRACE=1 logs every submit and present, to see how the game splits a frame.
pub static TRACE: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NX_INTERP_TRACE").is_some());

/// NX_INTERP_KEEP=0: unblended uniforms are read from live game memory instead of the copy taken for the frame.
pub static KEEP: LazyLock<bool> = LazyLock::new(|| std::env::var("NX_INTERP_KEEP").map_or(true, |v| v != "0"));

/// NX_INTERP_AT_PRESENT=1 reads uniforms in guest memory when the frame is presented rather than when each draw ran.
pub static AT_PRESENT: LazyLock<bool> = LazyLock::new(|| std::env::var("NX_INTERP_AT_PRESENT").is_ok_and(|v| v != "0"));

pub static ON: LazyLock<bool> = LazyLock::new(|| std::env::var("NX_INTERP").is_ok_and(|v| v != "0"));

/// Largest uniform buffer kept per slot; bigger bindings are left alone.
const MAX_SLOT: u64 = 0x10000;

struct Slot {
    stage: u8,
    index: u8,
    addr: u64,
    size: u32,
    off: u32,
}

struct DrawUbos {
    program: (u64, u64),
    mesh: u64,
    target: u64,
    depth: bool,
    slots: Vec<Slot>,
}

/// The uniform bytes every draw of one frame read, in draw order. Bytes from inline updates are taken when the draw is
/// recorded, since they only exist then; buffers in guest memory are read when the frame is presented (`fill`), because
/// the game may still write them after submitting the commands, and the GPU only reads them later.
#[derive(Default)]
pub struct Capture {
    draws: Vec<DrawUbos>,
    bytes: Vec<u8>,
    /// Bytes already kept, by (address, size, inline update): an inline update can rewrite a buffer between two draws, and
    /// each version is kept separately.
    seen: HashMap<(u64, u32, usize), u32>,
    pending: Vec<(u64, u32, u32)>,
}

impl Capture {
    pub fn draws(&self) -> usize {
        self.draws.len()
    }

    /// `mesh` tells the draw's geometry apart from others with the same program, `target` is the address of the colour
    /// target it draws into and `depth` whether a depth buffer is bound; `slots` yields (stage, index, guest address,
    /// size, the inline update holding the bytes if any).
    pub fn record(
        &mut self,
        program: (u64, u64),
        mesh: u64,
        target: u64,
        depth: bool,
        slots: impl Iterator<Item = (u8, u8, u64, u64, Option<*const u8>)>,
    ) {
        let mut out = Vec::new();
        for (stage, index, addr, size, inline) in slots {
            if size == 0 || size > MAX_SLOT {
                continue;
            }
            let size = size as u32;
            let key = (addr, size, inline.map_or(0, |p| p as usize));
            let off = match self.seen.get(&key) {
                Some(&off) => off,
                None => {
                    let off = self.bytes.len() as u32;
                    match inline.or((!*AT_PRESENT).then_some(addr as *const u8)) {
                        Some(ptr) => self.bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, size as usize) }),
                        None => {
                            self.bytes.resize(self.bytes.len() + size as usize, 0);
                            self.pending.push((addr, size, off));
                        }
                    }
                    self.seen.insert(key, off);
                    off
                }
            };
            out.push(Slot { stage, index, addr, size, off });
        }
        self.draws.push(DrawUbos { program, mesh, target, depth, slots: out });
    }

    /// Reads the guest-memory buffers, once the frame is complete.
    pub fn fill(&mut self) {
        for (addr, size, off) in std::mem::take(&mut self.pending) {
            let src = unsafe { std::slice::from_raw_parts(addr as *const u8, size as usize) };
            self.bytes[off as usize..(off + size) as usize].copy_from_slice(src);
        }
    }

    fn data(&self, s: &Slot) -> &[u8] {
        &self.bytes[s.off as usize..(s.off + s.size) as usize]
    }
}

pub struct Start {
    pub st: GState,
    pub colors: Vec<Option<Box<TexBuilder>>>,
    pub depth: Option<Box<TexBuilder>>,
}

/// Everything one game frame submitted, between two presents. `calls` holds the command buffers its commands call,
/// taken when they ran: by the time of the replay the game may already be reusing their memory.
#[derive(Default)]
pub struct Frame {
    pub cmds: Vec<Arc<Vec<Cmd>>>,
    /// Guest memory the GPU itself wrote this frame (compute output, texture read-backs). Uniforms in there are read live
    /// by the replay: the bytes the CPU saw when the frame was recorded are from before the GPU wrote them.
    pub gpu_written: Vec<(u64, u64)>,
    pub calls: HashMap<u64, Arc<Vec<Cmd>>>,
    pub start: Option<Start>,
    pub cap: Capture,
}

impl Frame {
    pub fn keep_calls(&mut self, cmds: &[Cmd], depth: usize) {
        for c in cmds {
            if let Cmd::Call(handles) = c {
                for h in handles {
                    if depth < 8 && !self.calls.contains_key(h) {
                        if let Some(sub) = super::cmd::recording(*h) {
                            self.calls.insert(*h, sub.clone());
                            self.keep_calls(&sub, depth + 1);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Default)]
pub struct State {
    pub cur: Frame,
    pub prev: Option<Capture>,
    pub hold: Option<crate::gfx::Shared<crate::gfx::Texture>>,
    /// Per draw of the frame being replayed: (stage, slot, GPU address of the blended bytes).
    pub overrides: Vec<Vec<(u8, u8, u64)>>,
    pub draw_index: usize,
    /// The replayed frame's called command buffers.
    pub calls: HashMap<u64, Arc<Vec<Cmd>>>,
}

/// NX_INTERP_T: where between the last frame (0) and this one (1) the extra image sits, 0.5 by default. 1 replays the
/// real frame unchanged, which isolates replay problems from blending ones.
pub static T: LazyLock<f32> = LazyLock::new(|| std::env::var("NX_INTERP_T").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5));

#[derive(Default, Debug)]
pub struct Built {
    pub draws: usize,
    /// 2D draws (no depth buffer, or into the shown image itself), never blended.
    pub hud: usize,
    /// Bytes of unblended uniforms uploaded so the replay does not read live game memory.
    pub kept_bytes: usize,
    /// Bindings left to read live because the GPU wrote them.
    pub live: usize,
    pub paired: usize,
    pub slots: usize,
    pub words: usize,
    pub bytes: usize,
}

/// How far apart two draws' uniforms are, over the start of each shared slot (where world matrices live).
fn distance(prev: &Capture, p: &DrawUbos, cur: &Capture, d: &DrawUbos) -> f32 {
    let mut sum = 0.0;
    for s in &d.slots {
        let Some(q) = p.slots.iter().find(|q| (q.stage, q.index, q.size) == (s.stage, s.index, s.size)) else {
            sum += 64.0;
            continue;
        };
        let (a, b) = (prev.data(q), cur.data(s));
        for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)).take(64) {
            let (x, y) = (f32::from_le_bytes(x.try_into().unwrap()), f32::from_le_bytes(y.try_into().unwrap()));
            sum += if x == y { 0.0 } else if x.is_finite() && y.is_finite() { (x - y).abs().min(10.0) } else { 10.0 };
        }
    }
    sum
}

/// Pairs each draw with a draw of the same mesh in the last frame, or failing that the same program, searching around
/// where the last pair was so the two stay in step when draws appear or disappear. Among several candidates the one
/// with the closest uniforms wins, so a coin is paired with itself and not with the coin next to it.
fn align(prev: &Capture, cur: &Capture) -> Vec<Option<usize>> {
    const BEHIND: usize = 16;
    const AHEAD: usize = 96;
    const CANDIDATES: usize = 8;
    let mut used = vec![false; prev.draws.len()];
    let mut next = 0usize;
    cur.draws
        .iter()
        .map(|d| {
            let window = next.saturating_sub(BEHIND)..(next + AHEAD).min(prev.draws.len());
            let pick = |same: &dyn Fn(&DrawUbos) -> bool| -> Vec<usize> {
                window.clone().filter(|&k| !used[k] && same(&prev.draws[k])).take(CANDIDATES).collect()
            };
            let mut cands = pick(&|p| p.mesh == d.mesh);
            if cands.is_empty() {
                cands = pick(&|p| p.program == d.program);
            }
            if cands.is_empty() {
                // A long run of draws vanished: find the mesh further on and resync there.
                cands.extend((next..prev.draws.len()).find(|&k| !used[k] && prev.draws[k].mesh == d.mesh));
            }
            let k = match cands.len() {
                0 => return None,
                1 => cands[0],
                _ => *cands.iter().min_by(|&&a, &&b| distance(prev, &prev.draws[a], cur, d).total_cmp(&distance(prev, &prev.draws[b], cur, d))).unwrap(),
            };
            used[k] = true;
            next = k + 1;
            Some(k)
        })
        .collect()
}

/// Blends 32-bit words that read as ordinary floats in both frames and moved by a plausible amount; integers, flags,
/// handles and anything that jumped keep this frame's value. Two kinds of float are left alone too: whole numbers that
/// change (flipbook frames, indices), and small values that move by more than half a unit, which are almost always
/// texture scrolls wrapping around. Blending either one shows a wrong image for half a frame, seen as flicker.
/// Returns how many words were blended.
fn blend(prev: &[u8], cur: &[u8], t: f32, out: &mut [u8]) -> usize {
    let plain = |v: f32| v == 0.0 || (v.is_normal() && v.abs() < 1e7);
    let mut n = 0;
    for ((a, b), o) in prev.chunks_exact(4).zip(cur.chunks_exact(4)).zip(out.chunks_exact_mut(4)) {
        if a == b {
            continue;
        }
        let (x, y) = (f32::from_le_bytes(a.try_into().unwrap()), f32::from_le_bytes(b.try_into().unwrap()));
        if !plain(x) || !plain(y) || (y - x).abs() > 0.5 * x.abs().max(y.abs()) + 1.0 {
            continue;
        }
        let whole = x.fract() == 0.0 && y.fract() == 0.0;
        let wraps = x.abs() <= 1.0 && y.abs() <= 1.0 && (y - x).abs() > 0.5;
        if whole || wraps {
            continue;
        }
        o.copy_from_slice(&(x + (y - x) * t).to_le_bytes());
        n += 1;
    }
    n
}

/// Uniform bindings for a replay of `cur` at `t` of the way from `prev` (0) to `cur` (1), for every slot of every draw:
/// the blended bytes where the draw has a partner and they changed, otherwise the bytes as they were when the frame
/// drew. The replay never reads the game's live buffers, which by then may already hold the next frame, half written.
/// `alloc` places bytes in GPU memory and returns their address.
/// Only 3D draws are blended, told apart by a bound depth buffer: menus, the HUD and full-screen effects draw without
/// one, from dozens of identical quads that reshuffle as they animate and are too easily paired with the wrong
/// partner, and they gain nothing from blending. Draws straight into `screen`, the image that gets shown, are left too.
pub fn build(
    prev: &Capture,
    cur: &Capture,
    t: f32,
    screen: u64,
    gpu_written: &[(u64, u64)],
    mut alloc: impl FnMut(&[u8]) -> u64,
) -> (Vec<Vec<(u8, u8, u64)>>, Built) {
    let live = |s: &Slot| gpu_written.iter().any(|&(a, n)| s.addr < a + n && a < s.addr + s.size as u64);
    let pairs = align(prev, cur);
    // Keyed by where the bytes sit in the captures, which tells versions of one buffer apart.
    let mut mixed_cache: HashMap<(u32, u32), Option<u64>> = HashMap::new();
    let mut kept: HashMap<u32, u64> = HashMap::new();
    let mut built = Built { draws: cur.draws.len(), ..Default::default() };
    let mut out = Vec::with_capacity(cur.draws.len());
    for (d, pair) in cur.draws.iter().zip(&pairs) {
        let partner = match pair {
            _ if !d.depth || (d.target == screen && screen != 0) => {
                built.hud += 1;
                None
            }
            Some(k) => {
                built.paired += 1;
                Some(&prev.draws[*k])
            }
            None => None,
        };
        let mut ov = Vec::with_capacity(d.slots.len());
        for s in &d.slots {
            if live(s) {
                built.live += 1;
                continue;
            }
            let q = partner.and_then(|p| p.slots.iter().find(|q| (q.stage, q.index, q.size) == (s.stage, s.index, s.size)));
            let mixed = q.and_then(|q| {
                *mixed_cache.entry((s.off, q.off)).or_insert_with(|| {
                    let (a, b) = (prev.data(q), cur.data(s));
                    if a == b {
                        return None;
                    }
                    let mut mixed = b.to_vec();
                    let words = blend(a, b, t, &mut mixed);
                    if words == 0 {
                        return None;
                    }
                    built.words += words;
                    built.bytes += mixed.len();
                    Some(alloc(&mixed))
                })
            });
            let gpu = match mixed {
                Some(g) => {
                    built.slots += 1;
                    g
                }
                None if !*KEEP => continue,
                None => *kept.entry(s.off).or_insert_with(|| {
                    built.kept_bytes += s.size as usize;
                    alloc(cur.data(s))
                }),
            };
            ov.push((s.stage, s.index, gpu));
        }
        out.push(ov);
    }
    (out, built)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn blends_floats_and_keeps_the_rest() {
        let prev = [bytes(&[0.25, 10.5, 1.0]), 7u32.to_le_bytes().to_vec(), bytes(&[1.0e-30])].concat();
        let cur = [bytes(&[0.75, 12.5, 1.0]), 9u32.to_le_bytes().to_vec(), bytes(&[3.0e12])].concat();
        let mut out = cur.clone();
        assert_eq!(blend(&prev, &cur, 0.5, &mut out), 2);
        let f = |i: usize| f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((f(0), f(1), f(2)), (0.5, 11.5, 1.0));
        assert_eq!(u32::from_le_bytes(out[12..16].try_into().unwrap()), 9);
        assert_eq!(f(4), 3.0e12);
    }

    #[test]
    fn leaves_flipbook_frames_and_wrapping_scrolls_alone() {
        let prev = bytes(&[3.0, 0.9, 0.2]);
        let cur = bytes(&[4.0, 0.1, 0.3]);
        let mut out = cur.clone();
        assert_eq!(blend(&prev, &cur, 0.5, &mut out), 1);
        let f = |i: usize| f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!((f(0), f(1)), (4.0, 0.1));
        assert!((f(2) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn keeps_each_inline_version_of_a_buffer() {
        let (v1, v2) = (bytes(&[1.0]), bytes(&[2.0]));
        let mut c = Capture::default();
        c.record((1, 0), 1, 0, true, std::iter::once((0u8, 0u8, 0x1000u64, 4u64, Some(v1.as_ptr()))));
        c.record((1, 0), 1, 0, true, std::iter::once((0u8, 0u8, 0x1000u64, 4u64, Some(v2.as_ptr()))));
        assert_eq!((c.data(&c.draws[0].slots[0]), c.data(&c.draws[1].slots[0])), (&v1[..], &v2[..]));
    }

    #[test]
    fn pairs_draws_across_a_missing_one() {
        let cap = |programs: &[u64]| {
            let mut c = Capture::default();
            for &p in programs {
                c.record((p, 0), p, 0, true, std::iter::empty());
            }
            c
        };
        assert_eq!(align(&cap(&[1, 2, 3, 4]), &cap(&[1, 3, 4, 5])), vec![Some(0), Some(2), Some(3), None]);
    }

    #[test]
    fn pairs_same_mesh_draws_by_their_uniforms() {
        let coins = |data: &[Vec<u8>]| {
            let mut c = Capture::default();
            for d in data {
                c.record((7, 0), 42, 0, true, std::iter::once((0u8, 0u8, d.as_ptr() as u64, 12u64, None)));
            }
            c.fill();
            c
        };
        let at = |xs: &[f32]| xs.iter().map(|x| bytes(&[*x, 0.0, 0.0])).collect::<Vec<_>>();
        let (before, after) = (at(&[0.0, 5.0, 10.0]), at(&[5.1, 10.1]));
        // The first coin was collected: the remaining two must keep their own partners.
        assert_eq!(align(&coins(&before), &coins(&after)), vec![Some(1), Some(2)]);
    }

    #[test]
    fn keeps_pairing_after_a_long_run_of_vanished_draws() {
        let cap = |programs: Vec<u64>| {
            let mut c = Capture::default();
            for p in programs {
                c.record((p, 0), p, 0, true, std::iter::empty());
            }
            c
        };
        let prev: Vec<u64> = (0..300).collect();
        let cur: Vec<u64> = (0..10).chain(150..300).collect();
        let pairs = align(&cap(prev), &cap(cur.clone()));
        assert_eq!(pairs, cur.iter().map(|&p| Some(p as usize)).collect::<Vec<_>>());
    }
}
