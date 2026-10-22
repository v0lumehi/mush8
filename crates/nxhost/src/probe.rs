// Mush8 - probe.rs
//! Frame-rate experiments, all off unless their variable is set.
//! NX_SCAN_AT=<game s> lists every guest value that looks like a time step (1/60, 60.0, 16.67 ms, ...) in writable CPU memory
//! (GPU memory pools skipped) to work/probe-scan.tsv. NX_POKE_AT=<game s> scales them from then on (steps by NX_POKE_SCALE,
//! default 0.5, rates by its inverse), optionally only the hits of NX_POKE_KIND=step|rate|all (default step) whose index among
//! them is in NX_POKE=<lo>..<hi>, and until NX_POKE_UNTIL=<game s>.
//! NX_TICK_AT=<game s> finds the simulation's clocks (floats that grow by 1/60 s, 16.67 ms or 1 per frame), then halves every
//! step candidate on its own for a few frames and records which clocks slow down with it (work/probe-ticks.tsv, probe-cands.tsv).
//! With NX_WRITERS=<values watched at once, 1-4> it instead puts store watchpoints on a sample of those clocks and integer frame counters and lists the
//! code that writes them (work/probe-writers.tsv).
use crate::mem::{perm, state, ADDRESS_SPACE_END, PAGE};
use std::sync::{LazyLock, Mutex};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Step32,
    Rate32,
    Step64,
    Rate64,
}

const F32: &[(u32, Kind, &str)] = &[
    (0x3C88_8888, Kind::Step32, "1/60"),
    (0x3C88_8889, Kind::Step32, "1/60"),
    (0x3D08_8888, Kind::Step32, "1/30"),
    (0x3D08_8889, Kind::Step32, "1/30"),
    (0x4185_5555, Kind::Step32, "16.67ms"),
    (0x4185_5556, Kind::Step32, "16.67ms"),
    (0x4205_5555, Kind::Step32, "33.33ms"),
    (0x4205_5556, Kind::Step32, "33.33ms"),
    (0x4270_0000, Kind::Rate32, "60"),
    (0x41F0_0000, Kind::Rate32, "30"),
];

const F64: &[(u64, Kind, &str)] = &[
    (0x3F91_1111_1111_1111, Kind::Step64, "1/60"),
    (0x3FA1_1111_1111_1111, Kind::Step64, "1/30"),
    (0x4030_AAAA_AAAA_AAAB, Kind::Step64, "16.67ms"),
    (0x404E_0000_0000_0000, Kind::Rate64, "60"),
    (0x403E_0000_0000_0000, Kind::Rate64, "30"),
];

struct Hit {
    addr: u64,
    kind: Kind,
    original: u64,
    patched: u64,
}

impl Hit {
    fn wide(&self) -> bool {
        matches!(self.kind, Kind::Step64 | Kind::Rate64)
    }

    /// Writes `to` if the slot still holds `from`, so a value the game has replaced in the meantime is left alone.
    fn swap(&self, from: u64, to: u64) {
        unsafe {
            if self.wide() {
                let p = self.addr as *mut u64;
                if p.read_volatile() == from {
                    p.write_volatile(to);
                }
            } else {
                let p = self.addr as *mut u32;
                if p.read_volatile() == from as u32 {
                    p.write_volatile(to as u32);
                }
            }
        }
    }

    fn apply(&self) {
        self.swap(self.original, self.patched);
    }

    fn restore(&self) {
        self.swap(self.patched, self.original);
    }
}

fn env_f32(k: &str) -> Option<f32> {
    std::env::var(k).ok().and_then(|v| v.parse().ok())
}

static SCAN_AT: LazyLock<Option<f32>> = LazyLock::new(|| env_f32("NX_SCAN_AT"));
static POKE_AT: LazyLock<Option<f32>> = LazyLock::new(|| env_f32("NX_POKE_AT"));
static POKE_UNTIL: LazyLock<Option<f32>> = LazyLock::new(|| env_f32("NX_POKE_UNTIL"));
static TICK_AT: LazyLock<Option<f32>> = LazyLock::new(|| env_f32("NX_TICK_AT"));
static STATE: Mutex<(bool, Option<Vec<Hit>>)> = Mutex::new((false, None));
static TICKS: Mutex<Option<TickRun>> = Mutex::new(None);

/// Called once per presented frame.
pub fn frame(now: f32) {
    if TICK_AT.is_some_and(|t| now >= t) {
        TICKS.lock().unwrap().get_or_insert_with(TickRun::start).step();
    }
    if SCAN_AT.is_none() && POKE_AT.is_none() {
        return;
    }
    let mut s = STATE.lock().unwrap();
    if !s.0 && SCAN_AT.is_some_and(|t| now >= t) {
        s.0 = true;
        let hits = scan();
        report(&hits);
    }
    if POKE_AT.is_some_and(|t| now >= t) {
        if POKE_UNTIL.is_some_and(|t| now >= t) {
            if let Some(hits) = s.1.take() {
                hits.iter().for_each(Hit::restore);
                eprintln!("[probe] restored {} values at {now:.1} s", hits.len());
            }
            return;
        }
        let hits = s.1.get_or_insert_with(|| {
            let all = scan();
            report(&all);
            let hits = select(all);
            eprintln!("[probe] poking {} values from {now:.1} s on", hits.len());
            hits
        });
        for h in hits.iter() {
            h.apply();
        }
    }
}

fn select(hits: Vec<Hit>) -> Vec<Hit> {
    let kind = std::env::var("NX_POKE_KIND").unwrap_or_else(|_| "step".into());
    let mut hits: Vec<Hit> = hits
        .into_iter()
        .filter(|h| match kind.as_str() {
            "rate" => matches!(h.kind, Kind::Rate32 | Kind::Rate64),
            "all" => true,
            _ => matches!(h.kind, Kind::Step32 | Kind::Step64),
        })
        .collect();
    if let Some((lo, hi)) = std::env::var("NX_POKE").ok().and_then(|v| {
        let (a, b) = v.split_once("..")?;
        Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))
    }) {
        hits = hits.into_iter().enumerate().filter(|(i, _)| (lo..hi).contains(i)).map(|(_, h)| h).collect();
    }
    let scale = env_f32("NX_POKE_SCALE").unwrap_or(0.5);
    for h in &mut hits {
        h.patched = match h.kind {
            Kind::Step32 => (f32::from_bits(h.original as u32) * scale).to_bits() as u64,
            Kind::Rate32 => (f32::from_bits(h.original as u32) / scale).to_bits() as u64,
            Kind::Step64 => (f64::from_bits(h.original) * scale as f64).to_bits(),
            Kind::Rate64 => (f64::from_bits(h.original) / scale as f64).to_bits(),
        };
    }
    hits
}

/// Resident, writable CPU pages: no stacks, no GPU memory pools.
fn pages() -> Vec<u64> {
    let mut pools = crate::nvn::pool_ranges();
    pools.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, size) in pools {
        match merged.last_mut() {
            Some((_, end)) if start <= *end => *end = (*end).max(start + size),
            _ => merged.push((start, start + size)),
        }
    }
    let in_pool = |p: u64| {
        let i = merged.partition_point(|&(s, _)| s <= p);
        i > 0 && p < merged[i - 1].1
    };
    let regions = {
        let a = crate::kernel::aspace();
        let map = a.map.lock().unwrap();
        let mut out = Vec::new();
        let mut addr = 0;
        while addr < ADDRESS_SPACE_END {
            let (start, size, b) = map.query(addr);
            if b.perm & perm::RW == perm::RW && !matches!(b.state, state::FREE | state::INACCESSIBLE | state::STACK) {
                out.push((start, size));
            }
            addr = start + size;
        }
        out
    };
    let mut out = Vec::new();
    for (start, size) in regions {
        let mut v = vec![0 as libc::c_char; (size / PAGE) as usize];
        if unsafe { libc::mincore(start as *const libc::c_void, size as usize, v.as_mut_ptr()) } != 0 {
            continue;
        }
        out.extend(v.iter().enumerate().filter(|(_, b)| **b & 1 != 0).map(|(i, _)| start + i as u64 * PAGE).filter(|&p| !in_pool(p)));
    }
    out
}

fn page_words(page: u64) -> &'static [u32] {
    unsafe { std::slice::from_raw_parts(page as *const u32, (PAGE / 4) as usize) }
}

fn scan() -> Vec<Hit> {
    let t0 = std::time::Instant::now();
    let mut hits = Vec::new();
    let pages = pages();
    for &page in &pages {
        let words = page_words(page);
        for (j, &w) in words.iter().enumerate() {
            let addr = page + j as u64 * 4;
            if let Some(&(_, kind, _)) = F32.iter().find(|(v, _, _)| *v == w) {
                hits.push(Hit { addr, kind, original: w as u64, patched: 0 });
            }
            if j % 2 == 1 {
                let d = (w as u64) << 32 | words[j - 1] as u64;
                if let Some(&(_, kind, _)) = F64.iter().find(|(v, _, _)| *v == d) {
                    hits.push(Hit { addr: addr - 4, kind, original: d, patched: 0 });
                }
            }
        }
    }
    eprintln!(
        "[probe] scanned {} MB in {:.0} ms: {} candidates",
        pages.len() as u64 * PAGE >> 20,
        t0.elapsed().as_secs_f64() * 1e3,
        hits.len()
    );
    hits
}

fn label(h: &Hit) -> &'static str {
    match h.kind {
        Kind::Step32 | Kind::Rate32 => F32.iter().find(|(v, _, _)| *v as u64 == h.original).map_or("?", |e| e.2),
        Kind::Step64 | Kind::Rate64 => F64.iter().find(|(v, _, _)| *v == h.original).map_or("?", |e| e.2),
    }
}

fn place(addr: u64) -> String {
    let s = crate::diag::symbolize(addr);
    if !s.starts_with("0x") {
        return s;
    }
    let a = crate::kernel::aspace();
    if addr >= a.heap() && addr < a.alias() {
        format!("heap+{:#x}", addr - a.heap())
    } else if addr >= a.alias() && addr < a.stack() {
        format!("alias+{:#x}", addr - a.alias())
    } else {
        s
    }
}

fn region(addr: u64) -> &'static str {
    let p = place(addr);
    if p.starts_with("heap") {
        "heap"
    } else if p.starts_with("main") {
        "main"
    } else {
        "other"
    }
}

fn write(path: &str, body: String, what: &str) {
    let _ = std::fs::create_dir_all("work");
    match std::fs::write(path, body) {
        Ok(()) => eprintln!("[probe] {what} written to {path}"),
        Err(e) => eprintln!("[probe] {path}: {e}"),
    }
}

fn report(hits: &[Hit]) {
    let mut tsv = String::from("index\tplace\tkind\tvalue\n");
    for (i, h) in hits.iter().enumerate() {
        tsv += &format!("{i}\t{}\t{}\t{}\n", place(h.addr), if h.wide() { "f64" } else { "f32" }, label(h));
    }
    write("work/probe-scan.tsv", tsv, &format!("{} candidates", hits.len()));
}

/// Frames per measurement window, and frames to let a restored value settle before the next one is tested.
const WINDOW: u64 = 12;
const SETTLE: u64 = 4;
const SNAPSHOT_LIMIT: usize = 3 << 30;

#[derive(Clone, Copy, PartialEq)]
enum Unit {
    Seconds,
    Millis,
    Frames,
}

impl Unit {
    const ALL: [Unit; 3] = [Unit::Seconds, Unit::Millis, Unit::Frames];

    fn per_frame(self) -> f32 {
        match self {
            Unit::Seconds => 1.0 / 60.0,
            Unit::Millis => 1000.0 / 60.0,
            Unit::Frames => 1.0,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Unit::Seconds => "s",
            Unit::Millis => "ms",
            Unit::Frames => "frames",
        }
    }
}

struct Tick {
    addr: u64,
    unit: Unit,
    last: f32,
    slowed_by: Vec<usize>,
}

impl Tick {
    fn read(&self) -> f32 {
        unsafe { (self.addr as *const f32).read_volatile() }
    }

    /// Growth over the last window relative to a full-speed clock: 1.0 untouched, 0.5 when it runs at half speed.
    fn ratio(&mut self, frames: u64) -> f32 {
        let now = self.read();
        let r = (now - self.last) / (self.unit.per_frame() * frames as f32);
        self.last = now;
        r
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Detect,
    Verify,
    All,
    Settle,
    Each,
    Writers,
    Done,
}

struct TickRun {
    phase: Phase,
    frame: u64,
    since: u64,
    pages: Vec<u64>,
    snapshot: Vec<u32>,
    ticks: Vec<Tick>,
    cands: Vec<Hit>,
    cur: usize,
    counters: Vec<(u64, u32)>,
    targets: Vec<(u64, &'static str)>,
    writers: std::collections::BTreeMap<(u64, u64), u32>,
}

fn slowed(r: f32) -> bool {
    (0.4..0.6).contains(&r)
}

fn steady(r: f32) -> bool {
    (0.85..1.15).contains(&r)
}

impl TickRun {
    fn start() -> Self {
        let t0 = std::time::Instant::now();
        let mut pages = pages();
        pages.truncate(SNAPSHOT_LIMIT / PAGE as usize);
        let mut snapshot = Vec::with_capacity(pages.len() * (PAGE / 4) as usize);
        for &p in &pages {
            snapshot.extend_from_slice(page_words(p));
        }
        eprintln!("[probe] tick snapshot of {} MB in {:.0} ms", snapshot.len() * 4 >> 20, t0.elapsed().as_secs_f64() * 1e3);
        TickRun {
            phase: Phase::Detect,
            frame: 0,
            since: 0,
            pages,
            snapshot,
            ticks: Vec::new(),
            cands: Vec::new(),
            cur: 0,
            counters: Vec::new(),
            targets: Vec::new(),
            writers: Default::default(),
        }
    }

    fn step(&mut self) {
        self.frame += 1;
        let elapsed = self.frame - self.since;
        match self.phase {
            Phase::Detect if elapsed >= WINDOW => {
                self.detect(elapsed);
                self.next(Phase::Verify);
            }
            Phase::Verify if elapsed >= WINDOW => {
                self.ticks.retain_mut(|t| steady(t.ratio(elapsed)));
                self.counters.retain_mut(|(a, last)| {
                    let now = unsafe { (*a as *const u32).read_volatile() };
                    let d = now.wrapping_sub(*last) as u64;
                    *last = now;
                    d + 1 >= elapsed && d <= elapsed + 1
                });
                self.summary();
                if std::env::var_os("NX_WRITERS").is_some() {
                    self.pick_targets();
                    self.next(Phase::Writers);
                    return;
                }
                self.cands = select(scan());
                eprintln!("[probe] halving all {} step candidates for {WINDOW} frames", self.cands.len());
                self.cands.iter().for_each(Hit::apply);
                self.next(Phase::All);
            }
            Phase::All => {
                self.cands.iter().for_each(Hit::apply);
                if elapsed >= WINDOW {
                    let rs: Vec<f32> = self.ticks.iter_mut().map(|t| t.ratio(elapsed)).collect();
                    let (s, n) = (rs.iter().filter(|r| slowed(**r)).count(), rs.iter().filter(|r| steady(**r)).count());
                    eprintln!("[probe] all halved: {s} clocks at half speed, {n} unchanged, {} other", rs.len() - s - n);
                    self.cands.iter().for_each(Hit::restore);
                    self.next(Phase::Settle);
                }
            }
            Phase::Settle if elapsed >= SETTLE => {
                if self.cur >= self.cands.len() {
                    self.finish();
                    self.next(Phase::Done);
                    return;
                }
                for t in &mut self.ticks {
                    t.last = t.read();
                }
                self.cands[self.cur].apply();
                self.next(Phase::Each);
            }
            Phase::Each => {
                self.cands[self.cur].apply();
                if elapsed >= WINDOW {
                    let cur = self.cur;
                    for t in &mut self.ticks {
                        if slowed(t.ratio(elapsed)) {
                            t.slowed_by.push(cur);
                        }
                    }
                    self.cands[cur].restore();
                    self.cur += 1;
                    if self.cur % 50 == 0 {
                        eprintln!("[probe] tested {}/{} candidates", self.cur, self.cands.len());
                    }
                    self.next(Phase::Settle);
                }
            }
            Phase::Writers => {
                // NX_WRITERS=<n> watches n values at a time (the CPU has at most four watchpoints).
                static GROUP: LazyLock<usize> =
                    LazyLock::new(|| std::env::var("NX_WRITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).clamp(1, 4));
                let (group_size, frames) = (*GROUP, 12 + 18 * (*GROUP as u64 - 1).min(1));
                let group: Vec<u64> = self.targets.iter().skip(self.cur * group_size).take(group_size).map(|t| t.0).collect();
                if group.is_empty() {
                    crate::diag::writers_watch(&[]);
                    self.finish_writers();
                    self.next(Phase::Done);
                    return;
                }
                crate::diag::writers_watch(&group);
                if elapsed >= frames {
                    crate::diag::writers_watch(&[]);
                    for (addr, pc) in crate::diag::writers_take() {
                        *self.writers.entry((addr, pc)).or_default() += 1;
                    }
                    self.cur += 1;
                    self.next(Phase::Writers);
                }
            }
            _ => {}
        }
    }

    fn pick_targets(&mut self) {
        fn spread<T: Copy>(v: &[T], n: usize) -> Vec<T> {
            if v.len() <= n {
                return v.to_vec();
            }
            (0..n).map(|i| v[i * v.len() / n]).collect()
        }
        let of = |u: Unit| self.ticks.iter().filter(|t| t.unit == u).map(|t| t.addr).collect::<Vec<_>>();
        let mut targets: Vec<(u64, &'static str)> = Vec::new();
        targets.extend(of(Unit::Seconds).into_iter().map(|a| (a, "s")));
        targets.extend(of(Unit::Millis).into_iter().map(|a| (a, "ms")));
        targets.extend(spread(&of(Unit::Frames), 60).into_iter().map(|a| (a, "frames")));
        let ints: Vec<u64> = self.counters.iter().map(|c| c.0).collect();
        targets.extend(spread(&ints, 60).into_iter().map(|a| (a, "int")));
        eprintln!("[probe] tracing writers of {} values ({} integer counters verified)", targets.len(), self.counters.len());
        self.targets = targets;
        self.cur = 0;
    }

    fn finish_writers(&self) {
        let mut tsv = String::from("target	unit	writer	hits
");
        let mut by_pc: std::collections::BTreeMap<u64, (u32, std::collections::BTreeSet<&'static str>, usize)> = Default::default();
        let mut silent = 0;
        for &(addr, unit) in &self.targets {
            let mut any = false;
            for (&(a, pc), &n) in self.writers.range((addr, 0)..=(addr, u64::MAX)) {
                debug_assert_eq!(a, addr);
                any = true;
                tsv += &format!("{}	{unit}	{}	{n}
", place(addr), place(pc));
                let e = by_pc.entry(pc).or_default();
                e.0 += n;
                e.1.insert(unit);
                e.2 += 1;
            }
            silent += !any as usize;
        }
        write("work/probe-writers.tsv", tsv, &format!("writers of {} values", self.targets.len()));
        let mut rows: Vec<_> = by_pc.iter().collect();
        rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.2));
        eprintln!("[probe] {} distinct writing instructions, {silent} of {} values saw no write", rows.len(), self.targets.len());
        for (pc, (n, units, values)) in rows.iter().take(25) {
            let units: Vec<&str> = units.iter().copied().collect();
            eprintln!("[probe] writer {} writes {values} values ({}), {n} hits", place(**pc), units.join("+"));
        }
    }

    fn next(&mut self, phase: Phase) {
        self.phase = phase;
        self.since = self.frame;
    }

    fn detect(&mut self, frames: u64) {
        let snapshot = std::mem::take(&mut self.snapshot);
        let mut counters = 0usize;
        let a = crate::kernel::aspace();
        for (k, &page) in std::mem::take(&mut self.pages).iter().enumerate() {
            if !a.readable(page, PAGE) {
                continue;
            }
            let old = &snapshot[k * (PAGE / 4) as usize..(k + 1) * (PAGE / 4) as usize];
            for (j, (&o, &n)) in old.iter().zip(page_words(page)).enumerate() {
                if n.wrapping_sub(o) as u64 + 1 >= frames && n.wrapping_sub(o) as u64 <= frames + 1 {
                    counters += 1;
                    self.counters.push((page + j as u64 * 4, n));
                }
                let (of, nf) = (f32::from_bits(o), f32::from_bits(n));
                if !of.is_finite() || !nf.is_finite() || nf <= of {
                    continue;
                }
                let d = nf - of;
                if let Some(unit) = Unit::ALL.into_iter().find(|u| steady(d / (u.per_frame() * frames as f32))) {
                    self.ticks.push(Tick { addr: page + j as u64 * 4, unit, last: nf, slowed_by: Vec::new() });
                }
            }
        }
        eprintln!("[probe] {} float clock candidates, {counters} integer per-frame counters", self.ticks.len());
    }

    fn summary(&self) {
        for unit in Unit::ALL {
            let of: Vec<&Tick> = self.ticks.iter().filter(|t| t.unit == unit).collect();
            let (heap, main) = (of.iter().filter(|t| region(t.addr) == "heap").count(), of.iter().filter(|t| region(t.addr) == "main").count());
            eprintln!("[probe] clocks in {:6}: {:6} (heap {heap}, main {main})", unit.name(), of.len());
        }
    }

    fn finish(&self) {
        let mut tsv = String::from("place\tunit\tslowed_by\n");
        let mut free = 0;
        for t in &self.ticks {
            free += t.slowed_by.is_empty() as usize;
            let by: Vec<String> = t.slowed_by.iter().map(|i| i.to_string()).collect();
            tsv += &format!("{}\t{}\t{}\n", place(t.addr), t.unit.name(), by.join(","));
        }
        write("work/probe-ticks.tsv", tsv, &format!("{} clocks", self.ticks.len()));
        let mut tsv = String::from("index\tplace\tvalue\tclocks_slowed\n");
        let mut rows: Vec<(usize, usize)> =
            (0..self.cands.len()).map(|i| (i, self.ticks.iter().filter(|t| t.slowed_by.contains(&i)).count())).collect();
        for &(i, n) in &rows {
            tsv += &format!("{i}\t{}\t{}\t{n}\n", place(self.cands[i].addr), label(&self.cands[i]));
        }
        write("work/probe-cands.tsv", tsv, &format!("{} candidates", self.cands.len()));
        rows.sort_by_key(|r| std::cmp::Reverse(r.1));
        for &(i, n) in rows.iter().take(12).filter(|r| r.1 > 0) {
            eprintln!("[probe] candidate {i} at {} ({}) slows {n} clocks", place(self.cands[i].addr), label(&self.cands[i]));
        }
        eprintln!("[probe] {free} of {} clocks slowed by no single candidate", self.ticks.len());
    }
}
