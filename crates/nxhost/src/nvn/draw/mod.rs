// Mush8 - mod.rs
// Constantin Clerc (c22dev) - 0.1
mod copy;
mod exec;
mod interp;
mod shaders;
mod state;
mod stats;
mod textures;

use super::cmd::{self, Cmd, Draw, TexBuilder};
use super::objects::STATE;
use crate::gfx::{self, Attachments, Load, RenderEncoder, Shared};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use state::{Arena, GState, States, Targets};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{LazyLock, Mutex};

pub(crate) static TRACE_DRAWS: AtomicBool = AtomicBool::new(false);
static DUMP_NEXT: AtomicBool = AtomicBool::new(false);
static UPPER_LEFT: AtomicBool = AtomicBool::new(false);
static ZERO_TO_ONE: AtomicBool = AtomicBool::new(false);

pub fn set_origin_upper_left(v: bool) {
    UPPER_LEFT.store(v, Relaxed);
}
pub fn set_depth_zero_to_one(v: bool) {
    ZERO_TO_ONE.store(v, Relaxed);
}

#[derive(Default)]
struct Engine {
    st: GState,
    textures: textures::Textures,
    shaders: shaders::Shaders,
    states: States,
    arena: Arena,
    syncs: HashMap<u64, Shared<gfx::CommandBuffer>>,
    last_submit: Option<Shared<gfx::CommandBuffer>>,
    colors: Vec<Option<Box<TexBuilder>>>,
    depth: Option<Box<TexBuilder>>,
    overlays: BTreeMap<u64, (u64, u64, usize)>,
    target_first_frame: HashMap<textures::TexKey, u64>,
    copier: copy::Copier,
    kernels: HashMap<u64, Shared<ProtocolObject<dyn MTLComputePipelineState>>>,
    bank_tex: [[u64; 32]; 3],
    bank_gpu: [u64; 3],
    bank_ok: [bool; 3],
    sampled: HashMap<textures::TexKey, Shared<gfx::Texture>>,
    touched: Vec<(textures::TexKey, Shared<gfx::Texture>)>,
    clears: Vec<PendingClear>,
    interp: interp::State,
}

struct PendingClear {
    key: textures::TexKey,
    sub: (u32, u32),
    tex: Shared<gfx::Texture>,
    load: Load,
    stencil: bool,
}

impl PendingClear {
    fn is_depth(&self) -> bool {
        matches!(self.load, Load::ClearDepthStencil(..))
    }
}

impl Engine {
    fn forget_dead(&mut self) {
        let dead = std::mem::take(&mut *DEAD.lock().unwrap());
        if !dead.is_empty() {
            let gone = |k: &textures::TexKey| dead.iter().any(|&(lo, hi)| k.addr >= lo && k.addr < hi);
            self.textures.evict(&gone);
            self.target_first_frame.retain(|k, _| !gone(k));
        }
    }
}

static ENGINE: LazyLock<Mutex<Engine>> = LazyLock::new(Default::default);

struct Pass {
    enc: Retained<RenderEncoder>,
    targets: Targets,
    size: (u32, u32),
    heap: u64,
    chunks: usize,
    keys: Vec<(textures::TexKey, (u32, u32))>,
    fresh: bool,
    fixed: Option<EncState>,
}

#[derive(Clone, Copy)]
struct EncState {
    pipeline: usize,
    depth: usize,
    cull: u32,
    winding: u32,
    fill: u32,
    bias: [u32; 3],
    depth_clamp: bool,
    origin: [u32; 4],
    zrange: [u32; 2],
    scissor: [u32; 4],
    stencil: [u32; 2],
    blend: [u32; 4],
}

struct Replay<'a> {
    e: &'a mut Engine,
    cb: Retained<gfx::CommandBuffer>,
    pass: Option<Pass>,
    heap: Option<&'static gfx::HeapBuffer>,
    passes: u32,
    /// Replaying a finished frame to draw an in-between image: no writes back to guest memory, no compute.
    interp: bool,
    cur_draw: usize,
}

impl Replay<'_> {
    fn run(&mut self, cmds: &[Cmd], depth: usize) {
        for c in cmds {
            let st = &mut self.e.st;
            match c {
                Cmd::SetRenderTargets { colors, depth } => {
                    if self.pass.is_some() && self.e.colors == *colors && self.e.depth == *depth {
                        continue;
                    }
                    self.end_pass();
                    self.e.colors = colors.clone();
                    self.e.depth = depth.clone();
                }
                Cmd::ClearColor { index, rgba } => {
                    if !st.render_enable {
                        continue;
                    }
                    self.end_pass();
                    let Some(Some(b)) = self.e.colors.get(*index as usize).cloned() else { continue };
                    if textures::is_compressed(u32::from_le_bytes(b[0x30..0x34].try_into().unwrap())) {
                        continue;
                    }
                    let Some((key, t)) = self.e.textures.get(&b, true) else { continue };
                    if TRACE_DRAWS.load(Relaxed) {
                        eprintln!("[frame] clear color rt{index} {key:?} = {rgba:?}");
                    }
                    let c = rgba.map(|x| x as f64);
                    let sub = super::objects::view_subresource(&b);
                    self.defer_clear(PendingClear { key, sub, tex: t, load: Load::ClearColor(c), stencil: false });
                }
                Cmd::ClearDepthStencil { depth: z, stencil } => {
                    if !st.render_enable {
                        continue;
                    }
                    self.end_pass();
                    let Some(b) = self.e.depth.clone() else { continue };
                    let fmt = u32::from_le_bytes(b[0x30..0x34].try_into().unwrap());
                    let Some((key, t)) = self.e.textures.get(&b, true) else { continue };
                    let sub = super::objects::view_subresource(&b);
                    self.defer_clear(PendingClear { key, sub, tex: t, load: Load::ClearDepthStencil(*z as f64, *stencil), stencil: gfx::has_stencil(fmt) });
                }
                Cmd::Call(handles) => {
                    if depth < 8 {
                        for h in handles {
                            let sub = if self.interp { self.e.interp.calls.get(h).cloned() } else { cmd::recording(*h) };
                            if let Some(sub) = sub {
                                self.run(&sub, depth + 1);
                            }
                        }
                    }
                }
                Cmd::Program(p) => st.set_program(p),
                Cmd::VertexAttribs(v) => {
                    for (i, w) in v.iter().take(16).enumerate() {
                        st.attribs[i] = *w;
                    }
                }
                Cmd::VertexStreams(v) => {
                    for (i, s) in v.iter().take(16).enumerate() {
                        st.streams[i] = *s;
                    }
                }
                Cmd::VertexBuffer { index, addr, size } => {
                    if let Some(v) = st.vbufs.get_mut(*index as usize) {
                        *v = (*addr, *size);
                    }
                }
                Cmd::Uniform { stage, index, addr, size } => {
                    if let Some(u) = stage_index(*stage).and_then(|s| st.ubos[s].get_mut(*index as usize)) {
                        *u = (*addr, *size);
                    }
                }
                Cmd::Texture { stage, index, handle } => {
                    if let Some(t) = stage_index(*stage).and_then(|s| st.textures[s].get_mut(*index as usize)) {
                        *t = *handle;
                    }
                }
                Cmd::Storage { stage, index, addr, size } => {
                    if let Some(u) = stage_index(*stage).and_then(|s| st.storage[s].get_mut(*index as usize)) {
                        *u = (*addr, *size);
                    }
                }
                Cmd::UpdateUniform { addr, size, offset, data } => self.update_uniform(*addr, *size, *offset, data),
                Cmd::ColorState(v) => st.color_state = *v,
                Cmd::ChannelMask(v) => st.channel_mask = *v,
                Cmd::Blend(v) => st.blend[(*v & 7) as usize] = *v,
                Cmd::DepthStencil(v) => st.depth_stencil = *v,
                Cmd::Polygon(v) => st.polygon = *v,
                Cmd::Viewport(v) => st.viewport = Some(*v),
                Cmd::Scissor(v) => st.scissor = Some(*v),
                Cmd::BlendColor(v) => st.blend_color = *v,
                Cmd::StencilRef(f, v) => set_faces(&mut st.stencil_ref, *f, *v),
                Cmd::StencilValueMask(f, v) => set_faces(&mut st.stencil_read, *f, *v),
                Cmd::StencilMask(f, v) => set_faces(&mut st.stencil_write, *f, *v),
                Cmd::TexturePool(p) => st.texture_pool = *p,
                Cmd::SamplerPool(p) => st.sampler_pool = *p,
                Cmd::DepthClamp(v) => st.depth_clamp = *v,
                Cmd::RasterizerDiscard(v) => st.rasterizer_discard = *v,
                Cmd::ReportCounter { .. } if self.interp => {}
                Cmd::CopyTextureToBuffer { .. } if self.interp => {}
                // Uploads from guest memory: the texture already holds what the real frame copied, and the source may
                // already hold the next video frame or streamed data.
                Cmd::CopyBufferToTexture { .. } if self.interp => {}
                Cmd::Dispatch(_) if self.interp && !*interp::KEEP_COMPUTE => {}
                Cmd::ReportCounter { counter, addr } => {
                    static T0: LazyLock<std::time::Instant> = LazyLock::new(std::time::Instant::now);
                    let ns = T0.elapsed().as_nanos() as u64;
                    let value = if matches!(*counter, 0 | 16) { ns } else { 0 };
                    if *addr != 0 {
                        unsafe {
                            (*addr as *mut u64).write_unaligned(value);
                            ((*addr + 8) as *mut u64).write_unaligned(ns);
                        }
                    }
                }
                Cmd::DepthRange(n, f) => st.depth_range = (*n, *f),
                Cmd::PolygonOffset(v) => st.polygon_offset = *v,
                Cmd::CopyBufferToTexture { src, dst, region } => {
                    self.end_pass();
                    self.flush_clears();
                    self.copy_buffer_to_texture(*src, dst, *region);
                }
                Cmd::CopyTextureToTexture { src, src_region, dst, dst_region, flags } => {
                    self.end_pass();
                    self.flush_clears();
                    let r = match (self.e.textures.get(src, false), self.e.textures.get(dst, true)) {
                        (Some((_, s)), Some((_, d))) => {
                            let (ss, ds) = (super::objects::view_subresource(src), super::objects::view_subresource(dst));
                            self.e.copier.copy(&self.cb, &s, ss, *src_region, &d, ds, *dst_region, *flags)
                        }
                        _ => Err("copy of an unavailable texture"),
                    };
                    let _ = r;
                }
                Cmd::CopyTextureToBuffer { src, region, dst } => {
                    if *interp::ON {
                        let format = u32::from_le_bytes(src[0x30..0x34].try_into().unwrap());
                        let (bw, bh, bpb) = super::format_info(format);
                        let [_, _, _, w, h, d] = region.map(|v| v.max(0) as u64);
                        let size = w.div_ceil(bw as u64) * bpb as u64 * h.div_ceil(bh as u64) * d.max(1);
                        self.e.interp.cur.gpu_written.push((*dst, size));
                    }
                    self.end_pass();
                    self.flush_clears();
                    let format = u32::from_le_bytes(src[0x30..0x34].try_into().unwrap());
                    let r = match self.e.textures.get(src, false) {
                        Some((_, s)) => copy::copy_to_buffer(&self.cb, &s.0, super::objects::view_subresource(src), *region, format, *dst),
                        None => Err("copy of an unavailable texture"),
                    };
                    let _ = r;
                }
                Cmd::ClearTexture { dst, region, color, mask } => {
                    if !st.render_enable {
                        continue;
                    }
                    self.end_pass();
                    self.flush_clears();
                    let r = match self.e.textures.get(dst, true) {
                        Some((_, d)) => self.e.copier.clear(&self.cb, &d, super::objects::view_subresource(dst), *region, *color, *mask),
                        None => Err("clear of an unavailable texture"),
                    };
                    let _ = r;
                }
                Cmd::RenderEnable(v) => st.render_enable = *v,
                Cmd::Dispatch(groups) => {
                    if *interp::ON {
                        let written = self.e.st.storage.iter().flatten().filter(|s| s.0 != 0).copied().collect::<Vec<_>>();
                        self.e.interp.cur.gpu_written.extend(written);
                    }
                    self.end_pass();
                    self.flush_clears();
                    let r = self.dispatch(*groups);
                    let _ = r;
                }
                Cmd::Draw(d) => {
                    let _ = self.draw(d);
                }
            }
        }
    }

    fn copy_buffer_to_texture(&mut self, src: u64, b: &TexBuilder, region: [i32; 6]) {
        let Some((key, tex)) = self.e.textures.get(b, true) else { return };
        let (bw, bh, bpb) = super::format_info(key.format);
        let [x, y, z, w, h, d] = region.map(|v| v.max(0) as usize);
        if w == 0 || h == 0 {
            return;
        }
        let (layer, level) = super::objects::view_subresource(b);
        let row = w.div_ceil(bw as usize) * bpb as usize;
        let image = row * h.div_ceil(bh as usize);
        let layered = matches!(tex.0.textureType(), MTLTextureType::Type2DArray | MTLTextureType::TypeCube | MTLTextureType::TypeCubeArray);
        let len = image * d.max(1);
        let (buf, off) = match gfx::heap().filter(|h| h.gpu_addr(src).is_some() && h.gpu_addr(src + len as u64 - 1).is_some()) {
            Some(h) => (h.buffer.clone(), (src - h.base) as usize),
            None => {
                let data = unsafe { std::slice::from_raw_parts(src as *const u8, len) };
                let a = self.e.arena.alloc_bytes(data);
                (a.buffer, a.offset)
            }
        };
        let Some(enc) = self.cb.blitCommandEncoder() else { return };
        for i in 0..d.max(1) {
            let (slice, oz, depth) = if layered { (layer as usize + z + i, 0, 1) } else if i == 0 { (0, z, d.max(1)) } else { break };
            unsafe {
                enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                    &buf.0,
                    off + i * image,
                    row,
                    image,
                    MTLSize { width: w, height: h, depth },
                    &tex.0,
                    slice,
                    level as usize,
                    MTLOrigin { x, y, z: oz },
                )
            };
        }
        enc.endEncoding();
    }

    fn end_pass(&mut self) {
        if let Some(p) = self.pass.take() {
            p.enc.endEncoding();
            for (k, sub) in &p.keys {
                if let Some((_, src)) = self.e.textures.cached(k) {
                    for dst in self.e.textures.aliases_of(k) {
                        textures::copy_blocks(&self.cb, &src.0, &dst.0, Some(*sub));
                    }
                }
            }
            self.passes += 1;
            if self.passes % PASSES_PER_COMMIT == 0 {
                if let Some(next) = gfx::gpu().queue.0.commandBuffer() {
                    commit(&std::mem::replace::<Retained<gfx::CommandBuffer>>(&mut self.cb, next));
                }
            }
        }
    }

    fn update_uniform(&mut self, addr: u64, size: u64, offset: u64, data: &[u8]) {
        let size = size.clamp(offset + data.len() as u64, 0x10000);
        let a = self.e.arena.alloc(size as usize);
        let src = match self.e.overlays.get(&addr) {
            Some(&(osize, _, ptr)) if osize >= size => ptr as *const u8,
            _ => addr as *const u8,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(src, a.ptr, size as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), a.ptr.add(offset as usize), data.len());
        }
        self.e.overlays.insert(addr, (size, a.gpu, a.ptr as usize));
        if a.new_chunk {
            self.use_chunk(&a.buffer);
        }
    }

    fn use_chunk(&mut self, b: &Shared<state::Buffer>) {
        if let Some(p) = &mut self.pass {
            p.enc.useResource_usage_stages(ProtocolObject::from_ref(&*b.0), MTLResourceUsage::Read, MTLRenderStages::Vertex | MTLRenderStages::Fragment);
            p.chunks += 1;
        }
    }

    fn gpu(&self, addr: u64, overlay: bool) -> u64 {
        if overlay {
            if let Some((&base, &(size, gpu, _))) = self.e.overlays.range(..=addr).next_back() {
                if addr < base + size {
                    return gpu + (addr - base);
                }
            }
        }
        self.heap.and_then(|h| h.gpu_addr(addr)).unwrap_or_else(|| gfx::gpu().zeros.0.gpuAddress())
    }

    fn defer_clear(&mut self, c: PendingClear) {
        self.e.clears.retain(|p| !(p.key == c.key && p.sub == c.sub && p.is_depth() == c.is_depth()));
        self.e.clears.push(c);
    }

    fn take_clear(&mut self, key: textures::TexKey, sub: (u32, u32), depth: bool) -> Option<PendingClear> {
        let i = self.e.clears.iter().position(|p| p.key == key && p.sub == sub && p.is_depth() == depth)?;
        Some(self.e.clears.remove(i))
    }

    fn flush_clears(&mut self) {
        for c in std::mem::take(&mut self.e.clears) {
            let a = if c.is_depth() {
                Attachments { colors: vec![], depth: Some((c.tex, c.load, c.stencil)), subs: vec![], depth_sub: c.sub }
            } else {
                Attachments { colors: vec![Some((c.tex, c.load))], depth: None, subs: vec![c.sub], depth_sub: (0, 0) }
            };
            gfx::run_pass(&self.cb, &a);
        }
    }

    fn ensure_pass(&mut self) -> Result<(), &'static str> {
        if self.pass.is_none() {
            let mut colors = Vec::new();
            let mut subs = Vec::new();
            let mut fmts = Vec::new();
            let mut size = None;
            let mut bound = Vec::new();
            for b in self.e.colors.clone() {
                let b = b.filter(|b| !textures::is_compressed(u32::from_le_bytes(b[0x30..0x34].try_into().unwrap())));
                let sub = b.as_ref().map_or((0, 0), |b| super::objects::view_subresource(b));
                let t = b.and_then(|b| self.e.textures.get(&b, true));
                let t = t.filter(|(k, _)| {
                    let dup = bound.contains(&(*k, sub));
                    bound.push((*k, sub));
                    !dup
                });
                fmts.push(t.as_ref().map(|(_, t)| t.0.pixelFormat()));
                colors.push(t.map(|(k, t)| {
                    if !self.e.touched.iter().any(|(tk, _)| *tk == k) {
                        self.e.touched.push((k, t.clone()));
                    }
                    size.get_or_insert(((k.width >> sub.1).max(1), (k.height >> sub.1).max(1)));
                    let load = self.take_clear(k, sub, false).map_or(Load::Keep, |c| c.load);
                    (t, load)
                }));
                subs.push(sub);
            }
            let mut depth = None;
            let mut dfmt = None;
            let mut stencil = false;
            let mut depth_sub = (0, 0);
            if let Some(b) = self.e.depth.clone() {
                if let Some((k, t)) = self.e.textures.get(&b, true) {
                    depth_sub = super::objects::view_subresource(&b);
                    stencil = gfx::has_stencil(k.format);
                    dfmt = Some(t.0.pixelFormat());
                    size.get_or_insert(((k.width >> depth_sub.1).max(1), (k.height >> depth_sub.1).max(1)));
                    let load = self.take_clear(k, depth_sub, true).map_or(Load::Keep, |c| c.load);
                    depth = Some((t, load, stencil));
                }
            }
            let Some(size) = size else { return Err("no render target") };
            let keys: Vec<(textures::TexKey, (u32, u32))> =
                self.e.colors.iter().flatten().filter_map(|b| Some((textures::TexKey::of(b)?, super::objects::view_subresource(b)))).collect();
            let frame = self.e.textures.frame;
            let mut fresh = false;
            for b in self.e.colors.iter().flatten().chain(self.e.depth.iter()) {
                if let Some(k) = textures::TexKey::of(b) {
                    let first = *self.e.target_first_frame.entry(k).or_insert(frame);
                    fresh |= first == frame;
                }
            }
            while fmts.last() == Some(&None) {
                fmts.pop();
            }
            self.flush_clears();
            let enc = gfx::begin_pass(&self.cb, &Attachments { colors, depth, subs, depth_sub }).ok_or("no encoder")?;
            let chunks: Vec<_> = self.e.arena.used.clone();
            for c in &chunks {
                enc.useResource_usage_stages(ProtocolObject::from_ref(&*c.0), MTLResourceUsage::Read, MTLRenderStages::Vertex | MTLRenderStages::Fragment);
            }
            let z = &gfx::gpu().zeros.0;
            enc.useResource_usage_stages(ProtocolObject::from_ref(&**z), MTLResourceUsage::Read, MTLRenderStages::Vertex | MTLRenderStages::Fragment);
            self.pass = Some(Pass { enc, targets: Targets { colors: fmts, depth: dfmt, stencil }, size, heap: 0, chunks: chunks.len(), fresh, keys, fixed: None });
        }
        Ok(())
    }
}

fn stage_index(nvn_stage: u32) -> Option<usize> {
    match nvn_stage {
        0 => Some(0),
        1 => Some(1),
        5 => Some(2),
        _ => None,
    }
}

fn set_faces(v: &mut [u32; 2], faces: u32, value: u32) {
    if faces & 1 != 0 {
        v[0] = value;
    }
    if faces & 2 != 0 {
        v[1] = value;
    }
}

struct Debug {
    trace_frame: Option<u32>,
    trace_at: Option<f32>,
    trace_count: u32,
    dump_rt_at: Option<f32>,
    dump_frames: Vec<u32>,
    capture_file: bool,
}

static DEBUG: LazyLock<Debug> = LazyLock::new(|| {
    let var = |k: &str| std::env::var(k).ok();
    Debug {
        trace_frame: var("NX_TRACE_FRAME").and_then(|v| v.parse().ok()),
        trace_at: var("NX_TRACE_AT").and_then(|v| v.parse().ok()),
        trace_count: var("NX_TRACE_COUNT").and_then(|v| v.parse().ok()).unwrap_or(1),
        dump_rt_at: var("NX_DUMP_RT_AT").and_then(|v| v.parse().ok()),
        dump_frames: var("NX_DUMP_FRAME").map(|v| v.split(',').filter_map(|f| f.parse().ok()).collect()).unwrap_or_default(),
        capture_file: var("NX_NO_CAPTURE_FILE").is_none(),
    }
});

pub fn submit(handles: &[u64]) {
    gfx::wait_foreground();
    objc2::rc::autoreleasepool(|_| {
        let mut e = ENGINE.lock().unwrap();
        e.arena.recycle();
        e.forget_dead();
        e.bank_ok = [false; 3];
        e.textures.frame += 1;
        if *interp::ON {
            let e = &mut *e;
            if *interp::TRACE {
                let draws: usize = handles.iter().filter_map(|h| cmd::recording(*h)).map(|c| count_draws(&c, 0)).sum();
                eprintln!("[interp-trace] submit {} buffers, {draws} draws", handles.len());
            }
            if e.interp.cur.start.is_none() {
                e.interp.cur.start = Some(interp::Start { st: e.st, colors: e.colors.clone(), depth: e.depth.clone() });
                e.interp.draw_index = 0;
            }
            for c in handles.iter().filter_map(|h| cmd::recording(*h)) {
                e.interp.cur.keep_calls(&c, 0);
                e.interp.cur.cmds.push(c);
            }
        }
        let Some(cb) = gfx::gpu().queue.0.commandBuffer() else { return };
        let mut r = Replay { e: &mut e, cb, pass: None, heap: None, passes: 0, interp: false, cur_draw: 0 };
        let mut updated = BTreeMap::new();
        for h in handles {
            if let Some(cmds) = cmd::recording(*h) {
                collect_updates(&cmds, 0, &mut updated);
            }
        }
        for (&addr, &size) in &updated {
            let size = size.min(0x10000) as usize;
            let src = unsafe { std::slice::from_raw_parts(addr as *const u8, size) };
            let a = r.e.arena.alloc_bytes(src);
            r.e.overlays.insert(addr, (size as u64, a.gpu, a.ptr as usize));
        }
        if *stats::ON {
            stats::updates(updated.len(), updated.values().sum());
        }
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, Relaxed);
        static T0: LazyLock<std::time::Instant> = LazyLock::new(std::time::Instant::now);
        static TRACED_AT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let dbg = &*DEBUG;
        let at = dbg.trace_at.is_some_and(|t| T0.elapsed().as_secs_f32() >= t) && TRACED_AT.fetch_add(1, Relaxed) < dbg.trace_count;
        static CAPTURE_LEFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if dbg.capture_file && n % 32 == 0 {
            if let Ok(v) = std::fs::read_to_string("work/capture") {
                let _ = std::fs::remove_file("work/capture");
                CAPTURE_LEFT.store(v.trim().parse::<u32>().unwrap_or(1).max(1), Relaxed);
            }
        }
        let capture = CAPTURE_LEFT.load(Relaxed) > 0;
        if capture {
            CAPTURE_LEFT.fetch_sub(1, Relaxed);
            DUMP_NEXT.store(true, Relaxed);
        }
        TRACE_DRAWS.store(capture || dbg.trace_frame == Some(n) || at, Relaxed);
        for h in handles {
            if let Some(cmds) = cmd::recording(*h) {
                r.run(&cmds, 0);
            }
        }
        r.end_pass();
        r.flush_clears();
        for (&addr, &(size, _, ptr)) in &r.e.overlays {
            unsafe { std::ptr::copy_nonoverlapping(ptr as *const u8, addr as *mut u8, size as usize) };
        }
        let cb = r.cb;
        commit(&cb);
        static DUMPED_AT: AtomicBool = AtomicBool::new(false);
        let dump_at = dbg.dump_rt_at.is_some_and(|t| T0.elapsed().as_secs_f32() >= t) && !DUMPED_AT.swap(true, Relaxed);
        if DUMP_NEXT.swap(false, Relaxed) || dump_at || dbg.dump_frames.contains(&n) {
            cb.waitUntilCompleted();
            let _ = std::fs::create_dir_all("work/frame");
            for (i, (k, t)) in std::mem::take(&mut e.touched).into_iter().enumerate() {
                let path = format!("work/frame/{n:04}_{i:03}_{:x}_f{}_{}x{}.png", k.addr, k.format, k.width, k.height);
                gfx::dump_png(&t.0, &path);
            }
            let _ = std::fs::create_dir_all("work/frame/tex");
            for (k, t) in std::mem::take(&mut e.sampled) {
                gfx::dump_png(&t.0, &format!("work/frame/tex/{n:04}_{:x}_f{}_{}x{}.png", k.addr, k.format, k.width, k.height));
            }
            eprintln!("[gfx] frame {n} dumped to work/frame");
        }
        e.touched.clear();
        e.sampled.clear();
        let cb = Shared(cb);
        e.arena.retire(cb.clone());
        e.overlays.clear();
        e.last_submit = Some(cb);
    });
}

static DEAD: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

pub fn forget_memory(addr: u64, size: u64) {
    if size > 0 {
        DEAD.lock().unwrap().push((addr, addr + size));
    }
}

pub fn fence_sync(sync: u64) {
    let mut e = ENGINE.lock().unwrap();
    if let Some(cb) = e.last_submit.clone() {
        e.syncs.insert(sync, cb);
    }
}

pub use shaders::precompile;
pub use state::prewarm;

pub fn finish() {
    let cb = ENGINE.lock().unwrap().last_submit.clone();
    if let Some(cb) = cb {
        crate::kernel::blocking(|| cb.0.waitUntilCompleted());
    }
}

pub fn sync_wait(sync: u64) -> u64 {
    let cb = ENGINE.lock().unwrap().syncs.remove(&sync);
    match cb {
        Some(cb) => {
            crate::kernel::blocking(|| cb.0.waitUntilCompleted());
            1
        }
        None => 0,
    }
}

const PASSES_PER_COMMIT: u32 = 24;

fn commit(cb: &gfx::CommandBuffer) {
    if *FPS {
        let h = block2::RcBlock::new(|cb: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let cb = unsafe { cb.as_ref() };
            static LAST_END: Mutex<f64> = Mutex::new(0.0);
            let (start, end) = (cb.GPUStartTime(), cb.GPUEndTime());
            let mut last = LAST_END.lock().unwrap();
            GPU_NS.fetch_add(((end - start.max(*last)).max(0.0) * 1e9) as u64, Relaxed);
            *last = last.max(end);
        });
        unsafe { cb.addCompletedHandler(block2::RcBlock::as_ptr(&h)) };
    }
    cb.commit();
}

static FPS: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NX_FPS").is_some());
static GPU_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn count_fps() {
    static STATS: Mutex<Option<(std::time::Instant, std::time::Instant, u32, f32)>> = Mutex::new(None);
    if !*FPS {
        return;
    }
    let now = std::time::Instant::now();
    let mut s = STATS.lock().unwrap();
    let (start, last, n, worst) = s.get_or_insert((now, now, 0, 0.0));
    *worst = worst.max(now.duration_since(*last).as_secs_f32() * 1000.0);
    *last = now;
    *n += 1;
    if now.duration_since(*start).as_secs() >= 1 {
        let gpu = GPU_NS.swap(0, Relaxed) as f32 / 1e6 / *n as f32;
        eprintln!("[fps] {:.1} fps, worst frame {worst:.1} ms, GPU {gpu:.1} ms/frame", *n as f32 / now.duration_since(*start).as_secs_f32());
        *s = Some((now, now, 0, 0.0));
    }
}

/// NX_TRACE_PRESENT: the guest call chain that presents a frame. Its callers up the chain are the per-frame loop (simulation + draw).
#[inline(never)]
fn trace_present() {
    use std::collections::HashMap;
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("NX_TRACE_PRESENT").is_some());
    static SITES: Mutex<Option<(std::time::Instant, HashMap<u64, (u64, Vec<u64>)>)>> = Mutex::new(None);
    if !*ON {
        return;
    }
    let mut fp: u64;
    unsafe { std::arch::asm!("mov {}, x29", out(reg) fp) };
    let mut chain = Vec::new();
    let readable = |a: u64| {
        if a < 0x70_0000_0000 {
            return a != 0;
        }
        let (_, _, b) = crate::kernel::aspace().map.lock().unwrap().query(a);
        b.perm & crate::mem::perm::R != 0 && b.state != crate::mem::state::FREE
    };
    let mut prev = 0u64;
    for _ in 0..40 {
        let same_stack = prev == 0 || (fp > prev && fp - prev < 1 << 20);
        if fp == 0 || fp % 16 != 0 || !(same_stack || readable(fp)) {
            break;
        }
        let (next, lr) = unsafe { (*(fp as *const u64), *((fp + 8) as *const u64)) };
        chain.push(lr);
        if next == fp {
            break;
        }
        prev = fp;
        fp = next;
    }
    let key = chain.iter().fold(0u64, |a, v| a.rotate_left(5) ^ v);
    let mut g = SITES.lock().unwrap();
    let (t0, map) = g.get_or_insert_with(|| (std::time::Instant::now(), HashMap::new()));
    map.entry(key).or_insert((0, chain)).0 += 1;
    if t0.elapsed().as_secs() >= 10 {
        let secs = t0.elapsed().as_secs_f64();
        let mut rows: Vec<_> = map.values().collect();
        rows.sort_by_key(|v| std::cmp::Reverse(v.0));
        for (n, bt) in rows.iter().take(4) {
            let bt: Vec<String> = bt.iter().map(|a| format!("{a:#x}")).collect();
            eprintln!("[present] {:6.1}/s  chain={}", *n as f64 / secs, bt.join(" "));
        }
        *g = None;
    }
}

pub fn present(texture: u64) {
    trace_present();
    count_fps();
    if *stats::ON {
        stats::frame_end();
    }
    objc2::rc::autoreleasepool(|_| {
        let b = super::objects::texture(texture).map(|t| t.builder);
        let Some(b) = b else { return };
        let (t, real) = {
            let mut e = ENGINE.lock().unwrap();
            e.forget_dead();
            let t = e.textures.get(&b, true);
            let real = match &t {
                Some((k, t)) if *interp::ON => {
                    if *interp::TRACE {
                        eprintln!("[interp-trace] present {:#x} after {} draws", k.addr, e.interp.cur.cap.draws());
                    }
                    interpolate(&mut e, t, k.addr)
                }
                _ => None,
            };
            (t, real)
        };
        if let Some((_, t)) = t {
            let flip = !UPPER_LEFT.load(Relaxed);
            match real {
                Some((real, true)) => {
                    dump_at(&real);
                    dump_interp(&t, &real);
                    crate::kernel::blocking(|| gfx::present_frames(&[&t.0, &real.0], flip));
                }
                Some((real, false)) => {
                    dump_at(&real);
                    crate::kernel::blocking(|| gfx::present(&real.0, flip));
                }
                None => {
                    dump_at(&t);
                    crate::kernel::blocking(|| gfx::present(&t.0, flip));
                }
            }
        }
    });
}

/// Replays the frame that just finished halfway between the last frame and this one, into the game's own render targets.
/// The real image is copied aside first and returned, to be shown after the in-between one now in `window`; `false` with
/// it means the replay went wrong and only the real image may be shown.
fn interpolate(e: &mut Engine, window: &Shared<gfx::Texture>, screen: u64) -> Option<(Shared<gfx::Texture>, bool)> {
    let frame = std::mem::take(&mut e.interp.cur);
    let prev = e.interp.prev.take();
    let mut cap = frame.cap;
    cap.fill();
    let result = (|| {
        let (prev, start) = (prev.as_ref()?, frame.start?);
        if frame.cmds.is_empty() || gfx::turbo() {
            return None;
        }
        let g = gfx::gpu();
        let w = &window.0;
        let fits = |h: &Shared<gfx::Texture>| h.0.width() == w.width() && h.0.height() == w.height() && h.0.pixelFormat() == w.pixelFormat();
        if !e.interp.hold.as_ref().is_some_and(fits) {
            let d = unsafe { MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(w.pixelFormat(), w.width(), w.height(), false) };
            d.setStorageMode(MTLStorageMode::Private);
            d.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::RenderTarget);
            e.interp.hold = Some(Shared(g.device.0.newTextureWithDescriptor(&d)?));
        }
        let hold = e.interp.hold.clone()?;
        let cb = g.queue.0.commandBuffer()?;
        let blit = cb.blitCommandEncoder()?;
        unsafe { blit.copyFromTexture_toTexture(w, &hold.0) };
        blit.endEncoding();

        let arena = &mut e.arena;
        let (overrides, built) = interp::build(prev, &cap, *interp::T, screen, &frame.gpu_written, |bytes| arena.alloc_bytes(bytes).gpu);
        static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if LOGGED.fetch_add(1, Relaxed) % 600 == 0 {
            let o: Vec<usize> = interp::OUTCOMES.iter().map(|c| c.swap(0, Relaxed)).collect();
            eprintln!(
                "[interp] {} draws ({} 2D), {} paired, {} uniform buffers blended ({} words, {} KB, {} KB kept as drawn, {} read live) | last replays: {} blended bindings, draws ran {} / shader wait {} / failed {}",
                built.draws,
                built.hud,
                built.paired,
                built.slots,
                built.words,
                built.bytes >> 10,
                built.kept_bytes >> 10,
                built.live,
                interp::BOUND.swap(0, Relaxed),
                o[0],
                o[1],
                o[2]
            );
        }
        e.interp.overrides = overrides;
        e.interp.draw_index = 0;
        let end = (e.st, std::mem::take(&mut e.colors), e.depth.take());
        e.st = start.st;
        e.colors = start.colors;
        e.depth = start.depth;
        e.bank_ok = [false; 3];
        let mut updated = BTreeMap::new();
        for c in &frame.cmds {
            collect_updates(c, 0, &mut updated);
        }
        e.interp.calls = frame.calls;
        let mut r = Replay { e: &mut *e, cb, pass: None, heap: None, passes: 0, interp: true, cur_draw: 0 };
        for (&addr, &size) in &updated {
            let size = size.min(0x10000) as usize;
            let a = r.e.arena.alloc_bytes(unsafe { std::slice::from_raw_parts(addr as *const u8, size) });
            r.e.overlays.insert(addr, (size as u64, a.gpu, a.ptr as usize));
        }
        for c in &frame.cmds {
            r.run(c, 0);
        }
        r.end_pass();
        r.flush_clears();
        let cb = r.cb;
        commit(&cb);
        // The replay reads the game's vertex, index and storage buffers too. A fence the game set on this frame must
        // also cover it, or the game rewrites those buffers for the next frame while the replay is still reading them.
        let cb = Shared(cb);
        if let Some(real) = e.last_submit.take() {
            let real = Retained::as_ptr(&real.0);
            for s in e.syncs.values_mut() {
                if Retained::as_ptr(&s.0) == real {
                    *s = cb.clone();
                }
            }
        }
        e.last_submit = Some(cb.clone());
        e.arena.retire(cb);
        e.overlays.clear();
        e.interp.overrides.clear();
        e.interp.calls.clear();
        (e.st, e.colors, e.depth) = end;
        e.bank_ok = [false; 3];
        // The replay must have walked the same draws in the same order, or the blended buffers went to the wrong ones.
        if e.interp.draw_index != cap.draws() {
            static MISMATCH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if MISMATCH.fetch_add(1, Relaxed) % 60 == 0 {
                eprintln!("[interp] replay drew {} draws, the frame {}: in-between image dropped", e.interp.draw_index, cap.draws());
            }
            return Some((hold, false));
        }
        Some((hold, true))
    })();
    e.interp.prev = Some(cap);
    result
}

/// NX_INTERP_DUMP=<game s>: saves the in-between and the real image of NX_INTERP_DUMP_N (3) frames from then on.
fn dump_interp(mid: &Shared<gfx::Texture>, real: &Shared<gfx::Texture>) {
    static AT: LazyLock<Option<f32>> = LazyLock::new(|| std::env::var("NX_INTERP_DUMP").ok().and_then(|v| v.parse().ok()));
    static COUNT: LazyLock<u32> = LazyLock::new(|| std::env::var("NX_INTERP_DUMP_N").ok().and_then(|v| v.parse().ok()).unwrap_or(3));
    static DONE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if !AT.is_some_and(|t| gfx::clock() >= t) || DONE.load(Relaxed) >= *COUNT {
        return;
    }
    let n = DONE.fetch_add(1, Relaxed);
    let _ = std::fs::create_dir_all("work/frame");
    gfx::dump_png(&mid.0, &format!("work/frame/interp{n:02}-a-mid.png"));
    gfx::dump_png(&real.0, &format!("work/frame/interp{n:02}-b-real.png"));
    eprintln!("[interp] frame {n} dumped");
}

pub fn write_texels(mut b: TexBuilder, view: u64, region: [i32; 6], data: *const u8, strides: Option<(usize, usize)>) {
    super::objects::apply_view(&mut b, view, true);
    objc2::rc::autoreleasepool(|_| {
        let mut e = ENGINE.lock().unwrap();
        e.forget_dead();
        let Some((key, tex)) = e.textures.get(&b, false) else { return };
        let (bw, bh, bpb) = super::format_info(key.format);
        let [x, y, z, w, h, d] = region.map(|v| v.max(0) as usize);
        if w == 0 || h == 0 || data.is_null() {
            return;
        }
        let row = w.div_ceil(bw as usize) * bpb as usize;
        let image = row * h.div_ceil(bh as usize);
        let (row, image) = strides.filter(|s| s.0 >= row).unwrap_or((row, image));
        let decoded = gfx::bc::soft(key.format).map(|k| {
            let blocks = row * h.div_ceil(4);
            (0..d.max(1)).flat_map(|i| gfx::bc::decode(k, unsafe { std::slice::from_raw_parts(data.add(i * image), blocks) }, row, w, h, 1).0).collect::<Vec<u8>>()
        });
        let (data, row, image) = match &decoded {
            Some(px) => (px.as_ptr(), px.len() / d.max(1) / h, px.len() / d.max(1)),
            None => (data, row, image),
        };
        let level = if view != 0 && unsafe { *(view as *const u8) } & 1 != 0 { (unsafe { *((view + 1) as *const u8) }) as usize } else { 0 };
        let layered = matches!(tex.0.textureType(), MTLTextureType::Type2DArray | MTLTextureType::TypeCube | MTLTextureType::TypeCubeArray);
        let slices = if layered { z..z + d.max(1) } else { 0..1 };
        if tex.0.storageMode() == MTLStorageMode::Private {
            let count = if layered { d.max(1) } else { 1 };
            let depth = if layered { 1 } else { d.max(1) };
            let len = image * (count * depth - 1) + row * h.div_ceil(bh as usize);
            let g = gfx::gpu();
            let Some(buf) = (unsafe { g.device.0.newBufferWithBytes_length_options(std::ptr::NonNull::new(data as *mut u8).unwrap().cast(), len, MTLResourceOptions::StorageModeShared) }) else { return };
            let Some(cb) = g.queue.0.commandBuffer() else { return };
            let Some(enc) = cb.blitCommandEncoder() else { return };
            for (i, slice) in slices.enumerate() {
                unsafe {
                    enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        &buf,
                        i * image,
                        row,
                        image,
                        MTLSize { width: w, height: h, depth },
                        &tex.0,
                        slice,
                        level,
                        MTLOrigin { x, y, z: if layered { 0 } else { z } },
                    )
                };
            }
            enc.endEncoding();
            cb.commit();
            return;
        }
        for (i, slice) in slices.enumerate() {
            let region = MTLRegion {
                origin: MTLOrigin { x, y, z: if layered { 0 } else { z } },
                size: MTLSize { width: w, height: h, depth: if layered { 1 } else { d.max(1) } },
            };
            let p = unsafe { data.add(i * image) } as *mut u8;
            unsafe {
                tex.0.replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    region,
                    level,
                    slice,
                    std::ptr::NonNull::new(p).unwrap().cast(),
                    row,
                    image,
                )
            };
        }
    });
}

pub fn read_texels(mut b: TexBuilder, view: u64, region: [i32; 6], data: *mut u8, strides: Option<(usize, usize)>) {
    super::objects::apply_view(&mut b, view, true);
    let [_, _, _, w, h, d] = region.map(|v| v.max(0) as usize);
    if w == 0 || h == 0 || data.is_null() {
        return;
    }
    finish();
    objc2::rc::autoreleasepool(|_| {
        let mut e = ENGINE.lock().unwrap();
        e.forget_dead();
        let Some((key, tex)) = e.textures.get(&b, true) else { return };
        drop(e);
        let (bw, bh, bpb) = super::format_info(key.format);
        let row = w.div_ceil(bw as usize) * bpb as usize;
        let image = row * h.div_ceil(bh as usize);
        let len = image * d.max(1);
        let g = gfx::gpu();
        let Some(buf) = g.device.0.newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared) else { return };
        let Some(cb) = g.queue.0.commandBuffer() else { return };
        if copy::copy_into(&cb, &tex.0, super::objects::view_subresource(&b), region, key.format, &buf, 0).is_err() {
            return;
        }
        cb.commit();
        crate::kernel::blocking(|| cb.waitUntilCompleted());
        let src = unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const u8, len) };
        let (dst_row, dst_image) = strides.filter(|s| s.0 >= row).unwrap_or((row, image));
        let rows = h.div_ceil(bh as usize);
        for z in 0..d.max(1) {
            for y in 0..rows {
                let from = &src[z * image + y * row..][..row];
                unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), data.add(z * dst_image + y * dst_row), row) };
            }
        }
    });
}

fn dump_at(t: &Shared<gfx::Texture>) {
    static TIMES: LazyLock<Mutex<Vec<f32>>> = LazyLock::new(|| {
        let mut v: Vec<f32> = std::env::var("NX_DUMP_AT").unwrap_or_default().split(',').filter_map(|x| x.trim().parse().ok()).collect();
        v.sort_by(|a, b| b.partial_cmp(a).unwrap());
        Mutex::new(v)
    });
    let now = gfx::clock();
    let mut times = TIMES.lock().unwrap();
    if times.last().is_some_and(|&at| now >= at) {
        let at = times.pop().unwrap();
        let _ = std::fs::create_dir_all("work/frame");
        gfx::dump_png(&t.0, &format!("work/frame/t{at:05.1}.png"));
        eprintln!("[gfx] screen at {at}s dumped");
    }
}

fn count_draws(cmds: &[Cmd], depth: usize) -> usize {
    cmds.iter()
        .map(|c| match c {
            Cmd::Draw(_) => 1,
            Cmd::Call(handles) if depth < 8 => handles.iter().filter_map(|h| cmd::recording(*h)).map(|s| count_draws(&s, depth + 1)).sum(),
            _ => 0,
        })
        .sum()
}

fn collect_updates(cmds: &[Cmd], depth: usize, out: &mut BTreeMap<u64, u64>) {
    for c in cmds {
        match c {
            Cmd::UpdateUniform { addr, size, offset, data } => {
                let s = out.entry(*addr).or_default();
                *s = (*s).max(*size).max(offset + data.len() as u64);
            }
            Cmd::Call(handles) if depth < 8 => {
                for h in handles {
                    if let Some(sub) = cmd::recording(*h) {
                        collect_updates(&sub, depth + 1, out);
                    }
                }
            }
            _ => {}
        }
    }
}
