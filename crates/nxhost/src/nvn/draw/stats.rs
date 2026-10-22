// Mush8 - stats.rs
//! NX_ISTAT=1: how stable the game's draw stream is from one frame to the next, the groundwork for drawing in-between
//! frames. Every 60 frames it compares the last two: draw order, which uniform buffers each draw reads, and whether
//! their contents changed.
use std::sync::{LazyLock, Mutex};

pub static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NX_ISTAT").is_some());

#[derive(Clone, PartialEq)]
struct Ubo {
    stage: u8,
    index: u8,
    addr: u64,
    size: u64,
    hash: u64,
    overlay: bool,
}

struct DrawRec {
    program: (u64, u64),
    ubos: Vec<Ubo>,
}

impl DrawRec {
    fn same_bindings(&self, o: &DrawRec) -> bool {
        self.program == o.program
            && self.ubos.len() == o.ubos.len()
            && self.ubos.iter().zip(&o.ubos).all(|(a, b)| (a.stage, a.index, a.addr, a.size) == (b.stage, b.index, b.addr, b.size))
    }
}

#[derive(Default)]
struct Frames {
    cur: Vec<DrawRec>,
    prev: Vec<DrawRec>,
    prev2: Vec<DrawRec>,
    prev3: Vec<DrawRec>,
    updates: (usize, u64),
    n: u64,
}

static FRAMES: Mutex<Option<Frames>> = Mutex::new(None);

pub fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

pub fn record(program: (u64, u64), ubos: impl Iterator<Item = (u8, u8, u64, u64, u64, bool)>) {
    let ubos = ubos.map(|(stage, index, addr, size, hash, overlay)| Ubo { stage, index, addr, size, hash, overlay }).collect();
    FRAMES.lock().unwrap().get_or_insert_with(Default::default).cur.push(DrawRec { program, ubos });
}

pub fn updates(count: usize, bytes: u64) {
    let mut g = FRAMES.lock().unwrap();
    let f = g.get_or_insert_with(Default::default);
    f.updates.0 += count;
    f.updates.1 += bytes;
}

pub fn frame_end() {
    let mut g = FRAMES.lock().unwrap();
    let f = g.get_or_insert_with(Default::default);
    f.n += 1;
    if f.n % 60 == 0 && !f.prev3.is_empty() {
        for (lag, prev) in [(1, &f.prev), (2, &f.prev2), (3, &f.prev3)] {
            report(lag, prev, &f.cur, f.updates);
        }
        report_slots(&f.prev, &f.prev2, &f.cur);
    }
    f.prev3 = std::mem::take(&mut f.prev2);
    f.prev2 = std::mem::take(&mut f.prev);
    f.prev = std::mem::take(&mut f.cur);
    f.updates = (0, 0);
}

fn pct(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}

/// Pairs each current draw with the next draw of the same program in `prev` (a short look-ahead keeps the walk in step).
fn align<'a>(prev: &'a [DrawRec], cur: &'a [DrawRec]) -> Vec<(&'a DrawRec, &'a DrawRec)> {
    let mut out = Vec::new();
    let mut j = 0;
    for d in cur {
        if let Some(k) = (j..(j + 64).min(prev.len())).find(|&k| prev[k].program == d.program) {
            out.push((d, &prev[k]));
            j = k + 1;
        }
    }
    out
}

/// Per uniform slot (stage, index): how often it keeps its address one and two frames later, and whether the bytes behind
/// it changed, i.e. which slots carry per-frame data such as matrices.
fn report_slots(prev: &[DrawRec], prev2: &[DrawRec], cur: &[DrawRec]) {
    #[derive(Default)]
    struct Slot {
        uses: usize,
        same1: usize,
        same2: usize,
        changed2: usize,
        sizes: std::collections::BTreeMap<u64, usize>,
    }
    let mut slots: std::collections::BTreeMap<(u8, u8), Slot> = Default::default();
    let a1 = align(prev, cur);
    let a2: std::collections::HashMap<*const DrawRec, &DrawRec> = align(prev2, cur).into_iter().map(|(c, p)| (c as *const _, p)).collect();
    for (c, p1) in &a1 {
        for u in &c.ubos {
            let s = slots.entry((u.stage, u.index)).or_default();
            s.uses += 1;
            *s.sizes.entry(u.size).or_default() += 1;
            s.same1 += p1.ubos.iter().any(|v| (v.stage, v.index, v.addr) == (u.stage, u.index, u.addr)) as usize;
            if let Some(p2) = a2.get(&(*c as *const _)) {
                if let Some(v) = p2.ubos.iter().find(|v| (v.stage, v.index) == (u.stage, u.index)) {
                    if v.addr == u.addr {
                        s.same2 += 1;
                        s.changed2 += (v.hash != u.hash) as usize;
                    }
                }
            }
        }
    }
    eprintln!("[istat] {} of {} draws paired with the last frame by program", a1.len(), cur.len());
    let mut rows: Vec<_> = slots.iter().collect();
    rows.sort_by_key(|(_, s)| std::cmp::Reverse(s.uses));
    for ((stage, index), s) in rows.into_iter().take(10) {
        let size = s.sizes.iter().max_by_key(|(_, n)| **n).map_or(0, |(sz, _)| *sz);
        eprintln!(
            "[istat]   slot {}{index}: {} uses, mostly {size} bytes | same address as frame -1 {:.0}%, as frame -2 {:.0}% | changed since -2 {:.0}% of those",
            if *stage == 0 { "v" } else { "f" },
            s.uses,
            pct(s.same1, s.uses),
            pct(s.same2, s.uses),
            pct(s.changed2, s.same2)
        );
    }
}

fn report(lag: u32, prev: &[DrawRec], cur: &[DrawRec], updates: (usize, u64)) {
    let same_pos = cur.iter().zip(prev).filter(|(a, b)| a.program == b.program).count();
    // Align the two streams: walk the current draws and look a little ahead in the previous frame for the same draw.
    let (mut j, mut aligned, mut aligned_prog, mut changed, mut changed_overlay, mut changed_memory) = (0usize, 0, 0, 0, 0, 0);
    for d in cur {
        let found = (j..(j + 64).min(prev.len())).find(|&k| prev[k].same_bindings(d));
        match found {
            Some(k) => {
                aligned += 1;
                let p = &prev[k];
                let diff: Vec<&Ubo> = d.ubos.iter().zip(&p.ubos).filter(|(a, b)| a.hash != b.hash).map(|(a, _)| a).collect();
                if !diff.is_empty() {
                    changed += 1;
                    changed_overlay += diff.iter().filter(|u| u.overlay).count();
                    changed_memory += diff.iter().filter(|u| !u.overlay).count();
                }
                j = k + 1;
            }
            None => aligned_prog += (j..(j + 64).min(prev.len())).any(|k| prev[k].program == d.program) as usize,
        }
    }
    let prev_addrs: std::collections::HashSet<(u64, u64)> = prev.iter().flat_map(|d| d.ubos.iter().map(|u| (u.addr, u.size))).collect();
    let cur_ubos: Vec<&Ubo> = cur.iter().flat_map(|d| d.ubos.iter()).collect();
    let reused = cur_ubos.iter().filter(|u| prev_addrs.contains(&(u.addr, u.size))).count();
    let overlay = cur_ubos.iter().filter(|u| u.overlay).count();
    eprintln!(
        "[istat] vs frame -{lag}: draws {} (then {}) | same program at same position {:.0}% | aligned with identical bindings {:.0}% (+{:.0}% same program, other buffers) | of aligned, uniforms changed {:.0}% ({} overlay, {} memory buffers)",
        cur.len(),
        prev.len(),
        pct(same_pos, cur.len()),
        pct(aligned, cur.len()),
        pct(aligned_prog, cur.len()),
        pct(changed, aligned),
        changed_overlay,
        changed_memory
    );
    eprintln!(
        "[istat] vs frame -{lag}: uniform bindings {} | address reused {:.0}% | from inline updates {:.0}% | updates {} ({} KB)",
        cur_ubos.len(),
        pct(reused, cur_ubos.len()),
        pct(overlay, cur_ubos.len()),
        updates.0,
        updates.1 >> 10
    );
}
