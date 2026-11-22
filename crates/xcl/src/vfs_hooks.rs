//! Process-wide Win32 file API hooks feeding the VFS.
//!
//! Hooks sit on KERNELBASE exports, so they also catch calls made by the CRT
//! (`fopen` → `CreateFileW`) and other system DLLs on the game's behalf – not just
//! the game's own imports. A thread-local guard makes the hooks transparent for any
//! file I/O the VFS itself performs.
//!
//! Known gap (documented in the report): direct `Nt*` file syscalls are not hooked.

use std::cell::Cell;
use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use windows_sys::Win32::Foundation::{SetLastError, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{WIN32_FIND_DATAA, WIN32_FIND_DATAW};

use crate::hooks::hook;
use crate::util::{from_ansi_ptr, from_wide_ptr, wide};
use crate::vfs::{self, Decision, Target};

thread_local! {
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

struct Guard;
impl Guard {
    fn enter() -> Option<Guard> {
        IN_HOOK.with(|c| {
            if c.get() {
                None
            } else {
                c.set(true);
                Some(Guard)
            }
        })
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        IN_HOOK.with(|c| c.set(false));
    }
}

macro_rules! orig {
    ($slot:ident as $ty:ty) => {
        std::mem::transmute::<usize, $ty>($slot.load(Ordering::Acquire))
    };
}

const INVALID_FILE_ATTRIBUTES: u32 = u32::MAX;

// ---------------------------------------------------------------------------------------
// CreateFile*
// ---------------------------------------------------------------------------------------

type FnCreateFileW = unsafe extern "system" fn(*const u16, u32, u32, *const c_void, u32, u32, HANDLE) -> HANDLE;
type FnCreateFileA = unsafe extern "system" fn(*const u8, u32, u32, *const c_void, u32, u32, HANDLE) -> HANDLE;
type FnCreateFile2 = unsafe extern "system" fn(*const u16, u32, u32, u32, *const c_void) -> HANDLE;
static O_CREATE_FILE_W: AtomicUsize = AtomicUsize::new(0);
static O_CREATE_FILE_A: AtomicUsize = AtomicUsize::new(0);
static O_CREATE_FILE_2: AtomicUsize = AtomicUsize::new(0);

/// The first opens with their flags (how a title reads: overlapped / unbuffered I/O writes
/// memory without the CPU – write-watch does not see it).
fn log_open(api: &str, path: &str, flags: u32) {
    static N: AtomicUsize = AtomicUsize::new(0);
    // the title's own data, not drivers' caches, system files or DLLs
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".dll") || lower.ends_with(".exe") || lower.contains(r"\appdata\") || lower.contains(r"\windows\") || lower.starts_with(r"\\") {
        return;
    }
    if N.fetch_add(1, Ordering::Relaxed) < 24 {
        crate::xlog_info!(
            "VFS: {api}({path}) flags {flags:#x}{}{}",
            if flags & 0x2000_0000 != 0 { " NO_BUFFERING" } else { "" },
            if flags & 0x4000_0000 != 0 { " OVERLAPPED" } else { "" }
        );
    }
}

unsafe extern "system" fn hk_create_file_w(n: *const u16, acc: u32, sh: u32, sa: *const c_void, disp: u32, fl: u32, t: HANDLE) -> HANDLE {
    let orig = orig!(O_CREATE_FILE_W as FnCreateFileW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, acc, sh, sa, disp, fl, t) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, acc, sh, sa, disp, fl, t) };
    log_open("CreateFileW", &path, fl);
    match v.open_decision(&path, acc, disp, fl) {
        Decision::Pass => orig(n, acc, sh, sa, disp, fl, t),
        Decision::Use(p) => orig(wide(&p).as_ptr(), acc, sh, sa, disp, fl, t),
        Decision::Fail(e) => {
            SetLastError(e);
            INVALID_HANDLE_VALUE
        }
    }
}

unsafe extern "system" fn hk_create_file_a(n: *const u8, acc: u32, sh: u32, sa: *const c_void, disp: u32, fl: u32, t: HANDLE) -> HANDLE {
    let orig = orig!(O_CREATE_FILE_A as FnCreateFileA);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, acc, sh, sa, disp, fl, t) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n, acc, sh, sa, disp, fl, t) };
    log_open("CreateFileA", &path, fl);
    match v.open_decision(&path, acc, disp, fl) {
        Decision::Pass => orig(n, acc, sh, sa, disp, fl, t),
        Decision::Use(p) => orig!(O_CREATE_FILE_W as FnCreateFileW)(wide(&p).as_ptr(), acc, sh, sa, disp, fl, t),
        Decision::Fail(e) => {
            SetLastError(e);
            INVALID_HANDLE_VALUE
        }
    }
}

unsafe extern "system" fn hk_create_file_2(n: *const u16, acc: u32, sh: u32, disp: u32, params: *const c_void) -> HANDLE {
    let orig = orig!(O_CREATE_FILE_2 as FnCreateFile2);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, acc, sh, disp, params) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, acc, sh, disp, params) };
    // CREATEFILE2_EXTENDED_PARAMETERS { dwSize, dwFileAttributes, dwFileFlags, ... }
    let flags = if params.is_null() { 0 } else { *((params as *const u8).add(8) as *const u32) };
    log_open("CreateFile2", &path, flags);
    match v.open_decision(&path, acc, disp, flags) {
        Decision::Pass => orig(n, acc, sh, disp, params),
        Decision::Use(p) => orig(wide(&p).as_ptr(), acc, sh, disp, params),
        Decision::Fail(e) => {
            SetLastError(e);
            INVALID_HANDLE_VALUE
        }
    }
}

// ---------------------------------------------------------------------------------------
// Attributes
// ---------------------------------------------------------------------------------------

type FnGetAttrW = unsafe extern "system" fn(*const u16) -> u32;
type FnGetAttrExW = unsafe extern "system" fn(*const u16, i32, *mut c_void) -> i32;
type FnSetAttrW = unsafe extern "system" fn(*const u16, u32) -> i32;
static O_GET_ATTR_W: AtomicUsize = AtomicUsize::new(0);
static O_GET_ATTR_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_GET_ATTR_A: AtomicUsize = AtomicUsize::new(0);
static O_GET_ATTR_EX_A: AtomicUsize = AtomicUsize::new(0);
static O_SET_ATTR_W: AtomicUsize = AtomicUsize::new(0);

/// None = pass through untouched; Some(Ok(path)) = use path; Some(Err(code)) = fail.
fn read_target(path: &str) -> Option<Result<String, u32>> {
    let v = vfs::get()?;
    match v.classify(path) {
        Target::Pass => None,
        Target::Translated(p) => Some(Ok(p)),
        Target::Game { rel } => Some(v.read_path_rel(&rel).ok_or(vfs::ERROR_FILE_NOT_FOUND)),
    }
}

unsafe extern "system" fn hk_get_attr_w(n: *const u16) -> u32 {
    let orig = orig!(O_GET_ATTR_W as FnGetAttrW);
    let Some(_g) = Guard::enter() else { return orig(n) };
    let Some(path) = from_wide_ptr(n) else { return orig(n) };
    match read_target(&path) {
        None => orig(n),
        Some(Ok(p)) => orig(wide(&p).as_ptr()),
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_FILE_ATTRIBUTES
        }
    }
}

unsafe extern "system" fn hk_get_attr_a(n: *const u8) -> u32 {
    type FnA = unsafe extern "system" fn(*const u8) -> u32;
    let orig = orig!(O_GET_ATTR_A as FnA);
    let Some(_g) = Guard::enter() else { return orig(n) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n) };
    match read_target(&path) {
        None => orig(n),
        Some(Ok(p)) => orig!(O_GET_ATTR_W as FnGetAttrW)(wide(&p).as_ptr()),
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_FILE_ATTRIBUTES
        }
    }
}

unsafe extern "system" fn hk_get_attr_ex_w(n: *const u16, lvl: i32, info: *mut c_void) -> i32 {
    let orig = orig!(O_GET_ATTR_EX_W as FnGetAttrExW);
    let Some(_g) = Guard::enter() else { return orig(n, lvl, info) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, lvl, info) };
    match read_target(&path) {
        None => orig(n, lvl, info),
        Some(Ok(p)) => orig(wide(&p).as_ptr(), lvl, info),
        Some(Err(e)) => {
            SetLastError(e);
            0
        }
    }
}

unsafe extern "system" fn hk_get_attr_ex_a(n: *const u8, lvl: i32, info: *mut c_void) -> i32 {
    type FnA = unsafe extern "system" fn(*const u8, i32, *mut c_void) -> i32;
    let orig = orig!(O_GET_ATTR_EX_A as FnA);
    let Some(_g) = Guard::enter() else { return orig(n, lvl, info) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n, lvl, info) };
    match read_target(&path) {
        None => orig(n, lvl, info),
        Some(Ok(p)) => orig!(O_GET_ATTR_EX_W as FnGetAttrExW)(wide(&p).as_ptr(), lvl, info),
        Some(Err(e)) => {
            SetLastError(e);
            0
        }
    }
}

unsafe extern "system" fn hk_set_attr_w(n: *const u16, attrs: u32) -> i32 {
    let orig = orig!(O_SET_ATTR_W as FnSetAttrW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, attrs) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, attrs) };
    match v.classify(&path) {
        Target::Pass => orig(n, attrs),
        Target::Translated(p) => orig(wide(&p).as_ptr(), attrs),
        Target::Game { rel } => {
            if !v.visible_in_overlay(&rel) && !v.visible_in_original(&rel) {
                SetLastError(vfs::ERROR_FILE_NOT_FOUND);
                return 0;
            }
            match v.prepare_write(&rel, true) {
                Ok(p) => orig(wide(&p).as_ptr(), attrs),
                Err(e) => {
                    SetLastError(e);
                    0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Free space: a console drive (T:\, G:\, …) reports the host volume its mount lives on.
// Untranslated, a console drive letter does not exist on the PC, the query fails and
// titles read it as a full disk (Minecraft then refuses to create a world).
// ---------------------------------------------------------------------------------------

type FnDiskFreeExW = unsafe extern "system" fn(*const u16, *mut u64, *mut u64, *mut u64) -> i32;
type FnDiskFreeExA = unsafe extern "system" fn(*const u8, *mut u64, *mut u64, *mut u64) -> i32;
type FnDiskFreeW = unsafe extern "system" fn(*const u16, *mut u32, *mut u32, *mut u32, *mut u32) -> i32;
type FnDiskFreeA = unsafe extern "system" fn(*const u8, *mut u32, *mut u32, *mut u32, *mut u32) -> i32;
static O_DISK_FREE_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_DISK_FREE_EX_A: AtomicUsize = AtomicUsize::new(0);
static O_DISK_FREE_W: AtomicUsize = AtomicUsize::new(0);
static O_DISK_FREE_A: AtomicUsize = AtomicUsize::new(0);

// A NULL directory (the current directory's volume) has no path to translate and passes through.

static SPACE_LOGGED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

/// What a title learned about free space, once per queried directory: titles act on these
/// figures (a failed or tiny answer reads as a full disk), so the log shows what they saw.
/// Call right after the API: the title's last-error value survives the logging.
fn log_space(api: &str, path: &str, real: Option<&str>, answer: Option<(u64, u64)>) {
    let err = unsafe { windows_sys::Win32::Foundation::GetLastError() };
    if let Ok(mut g) = SPACE_LOGGED.lock() {
        let seen = g.get_or_insert_with(HashSet::new);
        if seen.len() < 64 && seen.insert(format!("{api}|{}", path.to_lowercase())) {
            let host = match real {
                Some(r) if r != path => format!(" (host {r})"),
                _ => String::new(),
            };
            match answer {
                Some((avail, total)) => crate::xlog_info!("{api}({path}){host}: {} MiB available of {} MiB", avail >> 20, total >> 20),
                None => crate::xlog_warn!("{api}({path}){host} failed (error {err})"),
            }
        }
    }
    unsafe { SetLastError(err) };
}

/// `GetDiskFreeSpaceW` figures in bytes: (free clusters, total clusters) × bytes per cluster.
unsafe fn clusters_to_bytes(spc: *const u32, bps: *const u32, nfree: *const u32, ntotal: *const u32) -> Option<(u64, u64)> {
    if spc.is_null() || bps.is_null() || nfree.is_null() || ntotal.is_null() {
        return None;
    }
    let cluster = *spc as u64 * *bps as u64;
    Some((*nfree as u64 * cluster, *ntotal as u64 * cluster))
}

unsafe fn ex_answer(ok: i32, avail: *const u64, total: *const u64) -> Option<(u64, u64)> {
    (ok != 0).then(|| (avail.as_ref().copied().unwrap_or(0), total.as_ref().copied().unwrap_or(0)))
}

unsafe extern "system" fn hk_disk_free_ex_w(n: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32 {
    let orig = orig!(O_DISK_FREE_EX_W as FnDiskFreeExW);
    let Some(_g) = Guard::enter() else { return orig(n, avail, total, free) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, avail, total, free) };
    let (r, real) = match read_target(&path) {
        None => (orig(n, avail, total, free), None),
        Some(Ok(p)) => (orig(wide(&p).as_ptr(), avail, total, free), Some(p)),
        Some(Err(e)) => {
            SetLastError(e);
            (0, None)
        }
    };
    log_space("GetDiskFreeSpaceExW", &path, real.as_deref(), ex_answer(r, avail, total));
    r
}

unsafe extern "system" fn hk_disk_free_ex_a(n: *const u8, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32 {
    let orig = orig!(O_DISK_FREE_EX_A as FnDiskFreeExA);
    let Some(_g) = Guard::enter() else { return orig(n, avail, total, free) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n, avail, total, free) };
    let (r, real) = match read_target(&path) {
        None => (orig(n, avail, total, free), None),
        Some(Ok(p)) => (orig!(O_DISK_FREE_EX_W as FnDiskFreeExW)(wide(&p).as_ptr(), avail, total, free), Some(p)),
        Some(Err(e)) => {
            SetLastError(e);
            (0, None)
        }
    };
    log_space("GetDiskFreeSpaceExA", &path, real.as_deref(), ex_answer(r, avail, total));
    r
}

unsafe extern "system" fn hk_disk_free_w(n: *const u16, spc: *mut u32, bps: *mut u32, nfree: *mut u32, ntotal: *mut u32) -> i32 {
    let orig = orig!(O_DISK_FREE_W as FnDiskFreeW);
    let Some(_g) = Guard::enter() else { return orig(n, spc, bps, nfree, ntotal) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, spc, bps, nfree, ntotal) };
    let (r, real) = match read_target(&path) {
        None => (orig(n, spc, bps, nfree, ntotal), None),
        Some(Ok(p)) => (orig(wide(&p).as_ptr(), spc, bps, nfree, ntotal), Some(p)),
        Some(Err(e)) => {
            SetLastError(e);
            (0, None)
        }
    };
    let answer = if r != 0 { clusters_to_bytes(spc, bps, nfree, ntotal) } else { None };
    log_space("GetDiskFreeSpaceW", &path, real.as_deref(), answer);
    r
}

unsafe extern "system" fn hk_disk_free_a(n: *const u8, spc: *mut u32, bps: *mut u32, nfree: *mut u32, ntotal: *mut u32) -> i32 {
    let orig = orig!(O_DISK_FREE_A as FnDiskFreeA);
    let Some(_g) = Guard::enter() else { return orig(n, spc, bps, nfree, ntotal) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n, spc, bps, nfree, ntotal) };
    let (r, real) = match read_target(&path) {
        None => (orig(n, spc, bps, nfree, ntotal), None),
        Some(Ok(p)) => (orig!(O_DISK_FREE_W as FnDiskFreeW)(wide(&p).as_ptr(), spc, bps, nfree, ntotal), Some(p)),
        Some(Err(e)) => {
            SetLastError(e);
            (0, None)
        }
    };
    let answer = if r != 0 { clusters_to_bytes(spc, bps, nfree, ntotal) } else { None };
    log_space("GetDiskFreeSpaceA", &path, real.as_deref(), answer);
    r
}

// ---------------------------------------------------------------------------------------
// Directory / delete / move / copy
// ---------------------------------------------------------------------------------------

type FnCreateDirW = unsafe extern "system" fn(*const u16, *const c_void) -> i32;
type FnPathW = unsafe extern "system" fn(*const u16) -> i32;
type FnMoveFileExW = unsafe extern "system" fn(*const u16, *const u16, u32) -> i32;
type FnMoveFileWithProgressW = unsafe extern "system" fn(*const u16, *const u16, *const c_void, *const c_void, u32) -> i32;
type FnCopyFileExW = unsafe extern "system" fn(*const u16, *const u16, *const c_void, *const c_void, *mut i32, u32) -> i32;
type FnCopyFile2 = unsafe extern "system" fn(*const u16, *const u16, *const c_void) -> i32;
static O_CREATE_DIR_W: AtomicUsize = AtomicUsize::new(0);
static O_REMOVE_DIR_W: AtomicUsize = AtomicUsize::new(0);
static O_DELETE_FILE_W: AtomicUsize = AtomicUsize::new(0);
static O_MOVE_FILE_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_MOVE_FILE_PROGRESS_W: AtomicUsize = AtomicUsize::new(0);
static O_COPY_FILE_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_COPY_FILE_2: AtomicUsize = AtomicUsize::new(0);

unsafe fn bool_result(r: Result<(), u32>) -> i32 {
    match r {
        Ok(()) => 1,
        Err(e) => {
            SetLastError(e);
            0
        }
    }
}

unsafe extern "system" fn hk_create_dir_w(n: *const u16, sa: *const c_void) -> i32 {
    let orig = orig!(O_CREATE_DIR_W as FnCreateDirW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n, sa) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, sa) };
    match v.classify(&path) {
        Target::Pass => orig(n, sa),
        Target::Translated(p) => orig(wide(&p).as_ptr(), sa),
        Target::Game { rel } => bool_result(v.create_dir(&rel)),
    }
}

unsafe extern "system" fn hk_remove_dir_w(n: *const u16) -> i32 {
    let orig = orig!(O_REMOVE_DIR_W as FnPathW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n) };
    let Some(path) = from_wide_ptr(n) else { return orig(n) };
    match v.classify(&path) {
        Target::Pass => orig(n),
        Target::Translated(p) => orig(wide(&p).as_ptr()),
        Target::Game { rel } => bool_result(v.remove_dir(&rel)),
    }
}

unsafe extern "system" fn hk_delete_file_w(n: *const u16) -> i32 {
    let orig = orig!(O_DELETE_FILE_W as FnPathW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(n) };
    let Some(path) = from_wide_ptr(n) else { return orig(n) };
    match v.classify(&path) {
        Target::Pass => orig(n),
        Target::Translated(p) => orig(wide(&p).as_ptr()),
        Target::Game { rel } => bool_result(v.delete_file(&rel)),
    }
}

/// Source for a copy/move: (real path to read, game rel if it lives in the game folder).
fn source_of(v: &vfs::Vfs, path: &str) -> Result<(String, Option<String>), u32> {
    match v.classify(path) {
        Target::Pass => Ok((path.to_string(), None)),
        Target::Translated(p) => Ok((p, None)),
        Target::Game { rel } => {
            let p = v.read_path_rel(&rel).ok_or(vfs::ERROR_FILE_NOT_FOUND)?;
            Ok((p, Some(rel)))
        }
    }
}

fn dest_of(v: &vfs::Vfs, path: &str) -> Result<String, u32> {
    match v.classify(path) {
        Target::Pass => Ok(path.to_string()),
        Target::Translated(p) => {
            if let Some(parent) = std::path::Path::new(&p).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            Ok(p)
        }
        Target::Game { rel } => v.prepare_write(&rel, false),
    }
}

unsafe extern "system" fn hk_move_file_ex_w(src: *const u16, dst: *const u16, flags: u32) -> i32 {
    let orig = orig!(O_MOVE_FILE_EX_W as FnMoveFileExW);
    move_file(src, dst, flags, |s, d, f| orig(s, d, f))
}

/// MoveFileW, MoveFileA, MoveFileExA and MoveFileWithProgressA reach MoveFileWithProgressW
/// without going through MoveFileExW. (MoveFileExW may call it too: the guard then lets the
/// inner call through.)
unsafe extern "system" fn hk_move_file_with_progress_w(src: *const u16, dst: *const u16, prog: *const c_void, data: *const c_void, flags: u32) -> i32 {
    let orig = orig!(O_MOVE_FILE_PROGRESS_W as FnMoveFileWithProgressW);
    move_file(src, dst, flags, |s, d, f| orig(s, d, prog, data, f))
}

unsafe fn move_file(src: *const u16, dst: *const u16, flags: u32, orig: impl Fn(*const u16, *const u16, u32) -> i32) -> i32 {
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(src, dst, flags) };
    let (Some(s), d) = (from_wide_ptr(src), from_wide_ptr(dst)) else { return orig(src, dst, flags) };
    let Some(d) = d else { return orig(src, dst, flags) }; // delete-on-reboot form
    let s_game = matches!(v.classify(&s), Target::Game { .. });
    let d_game = matches!(v.classify(&d), Target::Game { .. });
    if !s_game && !d_game {
        let (rs, _) = match source_of(v, &s) {
            Ok(x) => x,
            Err(e) => return bool_result(Err(e)),
        };
        let rd = match dest_of(v, &d) {
            Ok(x) => x,
            Err(e) => return bool_result(Err(e)),
        };
        return orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), flags);
    }
    let r = (|| -> Result<(), u32> {
        let (rs, srel) = source_of(v, &s)?;
        let rd = dest_of(v, &d)?;
        match &srel {
            // source in the game folder: copy, then hide the source
            Some(rel) => {
                if std::path::Path::new(&rs).is_dir() {
                    return Err(vfs::ERROR_ACCESS_DENIED); // moving whole game dirs is not supported
                }
                std::fs::copy(&rs, &rd).map_err(|_| vfs::ERROR_ACCESS_DENIED)?;
                v.delete_file(rel).or_else(|e| if e == vfs::ERROR_FILE_NOT_FOUND { Ok(()) } else { Err(e) })?;
                Ok(())
            }
            None => {
                if orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), flags | 0x2 /* COPY_ALLOWED */) == 0 {
                    return Err(windows_sys::Win32::Foundation::GetLastError());
                }
                Ok(())
            }
        }
    })();
    bool_result(r)
}

unsafe extern "system" fn hk_copy_file_ex_w(src: *const u16, dst: *const u16, prog: *const c_void, data: *const c_void, cancel: *mut i32, flags: u32) -> i32 {
    let orig = orig!(O_COPY_FILE_EX_W as FnCopyFileExW);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(src, dst, prog, data, cancel, flags) };
    let (Some(s), Some(d)) = (from_wide_ptr(src), from_wide_ptr(dst)) else { return orig(src, dst, prog, data, cancel, flags) };
    let rs = match source_of(v, &s) {
        Ok((p, _)) => p,
        Err(e) => return bool_result(Err(e)),
    };
    let rd = match dest_of(v, &d) {
        Ok(p) => p,
        Err(e) => return bool_result(Err(e)),
    };
    orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), prog, data, cancel, flags)
}

unsafe extern "system" fn hk_copy_file_2(src: *const u16, dst: *const u16, params: *const c_void) -> i32 {
    let orig = orig!(O_COPY_FILE_2 as FnCopyFile2);
    let (Some(_g), Some(v)) = (Guard::enter(), vfs::get()) else { return orig(src, dst, params) };
    let (Some(s), Some(d)) = (from_wide_ptr(src), from_wide_ptr(dst)) else { return orig(src, dst, params) };
    let rs = match source_of(v, &s) {
        Ok((p, _)) => p,
        Err(e) => return crate::util::hresult_from_win32(e),
    };
    let rd = match dest_of(v, &d) {
        Ok(p) => p,
        Err(e) => return crate::util::hresult_from_win32(e),
    };
    orig(wide(&rs).as_ptr(), wide(&rd).as_ptr(), params)
}

// ---------------------------------------------------------------------------------------
// Directory enumeration (merges overlay + original, hides whiteouts)
// ---------------------------------------------------------------------------------------

type FnFindFirstExW = unsafe extern "system" fn(*const u16, i32, *mut c_void, i32, *const c_void, u32) -> HANDLE;
type FnFindFirstW = unsafe extern "system" fn(*const u16, *mut WIN32_FIND_DATAW) -> HANDLE;
type FnFindFirstA = unsafe extern "system" fn(*const u8, *mut WIN32_FIND_DATAA) -> HANDLE;
type FnFindNextW = unsafe extern "system" fn(HANDLE, *mut WIN32_FIND_DATAW) -> i32;
type FnFindNextA = unsafe extern "system" fn(HANDLE, *mut WIN32_FIND_DATAA) -> i32;
type FnFindClose = unsafe extern "system" fn(HANDLE) -> i32;
static O_FIND_FIRST_EX_W: AtomicUsize = AtomicUsize::new(0);
static O_FIND_FIRST_W: AtomicUsize = AtomicUsize::new(0);
static O_FIND_FIRST_A: AtomicUsize = AtomicUsize::new(0);
static O_FIND_NEXT_W: AtomicUsize = AtomicUsize::new(0);
static O_FIND_NEXT_A: AtomicUsize = AtomicUsize::new(0);
static O_FIND_CLOSE: AtomicUsize = AtomicUsize::new(0);

struct MergedFind {
    items: Vec<WIN32_FIND_DATAW>,
    pos: usize,
}

static MERGED: Mutex<Option<HashSet<usize>>> = Mutex::new(None);

fn is_merged(h: HANDLE) -> bool {
    MERGED.lock().ok().and_then(|g| g.as_ref().map(|s| s.contains(&(h as usize)))).unwrap_or(false)
}

fn name_of(d: &WIN32_FIND_DATAW) -> String {
    let n = d.cFileName.iter().position(|&c| c == 0).unwrap_or(d.cFileName.len());
    String::from_utf16_lossy(&d.cFileName[..n])
}

unsafe fn collect(pattern: &str) -> Vec<WIN32_FIND_DATAW> {
    let first = orig!(O_FIND_FIRST_EX_W as FnFindFirstExW);
    let next = orig!(O_FIND_NEXT_W as FnFindNextW);
    let close = orig!(O_FIND_CLOSE as FnFindClose);
    let mut out = Vec::new();
    let mut d: WIN32_FIND_DATAW = std::mem::zeroed();
    let h = first(wide(pattern).as_ptr(), 0, &mut d as *mut _ as *mut c_void, 0, std::ptr::null(), 0);
    if h == INVALID_HANDLE_VALUE {
        return out;
    }
    loop {
        out.push(d);
        if next(h, &mut d) == 0 {
            break;
        }
    }
    close(h);
    out
}

/// Some(handle) when the VFS handled the call; None to pass through.
unsafe fn find_first_vfs(path: &str, out: *mut WIN32_FIND_DATAW) -> Option<Result<HANDLE, u32>> {
    let v = vfs::get()?;
    match v.classify(path) {
        Target::Pass => None,
        Target::Translated(p) => {
            let first = orig!(O_FIND_FIRST_EX_W as FnFindFirstExW);
            let h = first(wide(&p).as_ptr(), 0, out as *mut c_void, 0, std::ptr::null(), 0);
            Some(if h == INVALID_HANDLE_VALUE { Err(windows_sys::Win32::Foundation::GetLastError()) } else { Ok(h) })
        }
        Target::Game { rel } => {
            let (dir_rel, pattern) = match rel.rsplit_once('\\') {
                Some((d, p)) => (d.to_string(), p.to_string()),
                None => (String::new(), rel.clone()),
            };
            let wildcard = pattern.contains('*') || pattern.contains('?');
            if !wildcard && !v.visible_in_overlay(&rel) && v.is_whiteout(&rel) {
                return Some(Err(vfs::ERROR_FILE_NOT_FOUND));
            }
            let join = |base: String| if pattern.is_empty() { base } else { format!("{base}\\{pattern}") };
            let mut items: Vec<WIN32_FIND_DATAW> = Vec::new();
            let mut seen: HashSet<String> = HashSet::new();
            if v.needs_merge(&dir_rel) || v.visible_in_overlay(&rel) {
                for d in collect(&join(v.overlay_path(&dir_rel))) {
                    let n = name_of(&d);
                    if n.eq_ignore_ascii_case(".xcl_whiteouts") {
                        continue;
                    }
                    if seen.insert(n.to_lowercase()) {
                        items.push(d);
                    }
                }
            }
            for d in collect(&join(v.original_path(&dir_rel))) {
                let n = name_of(&d);
                let child = if dir_rel.is_empty() { n.clone() } else { format!("{dir_rel}\\{n}") };
                if n != "." && n != ".." && v.is_whiteout(&child) {
                    continue;
                }
                if seen.insert(n.to_lowercase()) {
                    items.push(d);
                }
            }
            if items.is_empty() {
                return Some(Err(vfs::ERROR_FILE_NOT_FOUND));
            }
            *out = items[0];
            let h = Box::into_raw(Box::new(MergedFind { items, pos: 1 })) as HANDLE;
            MERGED.lock().unwrap().get_or_insert_with(HashSet::new).insert(h as usize);
            Some(Ok(h))
        }
    }
}

unsafe extern "system" fn hk_find_first_ex_w(n: *const u16, lvl: i32, data: *mut c_void, op: i32, filt: *const c_void, fl: u32) -> HANDLE {
    let orig = orig!(O_FIND_FIRST_EX_W as FnFindFirstExW);
    // only the common info levels/ops are virtualised
    if lvl > 1 || op != 0 {
        return orig(n, lvl, data, op, filt, fl);
    }
    let Some(_g) = Guard::enter() else { return orig(n, lvl, data, op, filt, fl) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, lvl, data, op, filt, fl) };
    match find_first_vfs(&path, data as *mut WIN32_FIND_DATAW) {
        None => orig(n, lvl, data, op, filt, fl),
        Some(Ok(h)) => h,
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_HANDLE_VALUE
        }
    }
}

unsafe extern "system" fn hk_find_first_w(n: *const u16, data: *mut WIN32_FIND_DATAW) -> HANDLE {
    let orig = orig!(O_FIND_FIRST_W as FnFindFirstW);
    let Some(_g) = Guard::enter() else { return orig(n, data) };
    let Some(path) = from_wide_ptr(n) else { return orig(n, data) };
    match find_first_vfs(&path, data) {
        None => orig(n, data),
        Some(Ok(h)) => h,
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_HANDLE_VALUE
        }
    }
}

unsafe fn w_to_a(w: &WIN32_FIND_DATAW, a: *mut WIN32_FIND_DATAA) {
    use windows_sys::Win32::Globalization::{WideCharToMultiByte, CP_ACP};
    let a = &mut *a;
    a.dwFileAttributes = w.dwFileAttributes;
    a.ftCreationTime = w.ftCreationTime;
    a.ftLastAccessTime = w.ftLastAccessTime;
    a.ftLastWriteTime = w.ftLastWriteTime;
    a.nFileSizeHigh = w.nFileSizeHigh;
    a.nFileSizeLow = w.nFileSizeLow;
    a.dwReserved0 = w.dwReserved0;
    a.dwReserved1 = w.dwReserved1;
    a.cFileName = [0; 260];
    a.cAlternateFileName = [0; 14];
    WideCharToMultiByte(CP_ACP, 0, w.cFileName.as_ptr(), -1, a.cFileName.as_mut_ptr() as *mut u8, 260, std::ptr::null(), std::ptr::null_mut());
    WideCharToMultiByte(CP_ACP, 0, w.cAlternateFileName.as_ptr(), -1, a.cAlternateFileName.as_mut_ptr() as *mut u8, 14, std::ptr::null(), std::ptr::null_mut());
}

unsafe extern "system" fn hk_find_first_a(n: *const u8, data: *mut WIN32_FIND_DATAA) -> HANDLE {
    let orig = orig!(O_FIND_FIRST_A as FnFindFirstA);
    let Some(_g) = Guard::enter() else { return orig(n, data) };
    let Some(path) = from_ansi_ptr(n) else { return orig(n, data) };
    let mut w: WIN32_FIND_DATAW = std::mem::zeroed();
    match find_first_vfs(&path, &mut w) {
        None => orig(n, data),
        Some(Ok(h)) => {
            w_to_a(&w, data);
            h
        }
        Some(Err(e)) => {
            SetLastError(e);
            INVALID_HANDLE_VALUE
        }
    }
}

unsafe fn merged_next(h: HANDLE) -> Option<WIN32_FIND_DATAW> {
    let m = &mut *(h as *mut MergedFind);
    let r = m.items.get(m.pos).copied();
    m.pos += 1;
    r
}

unsafe extern "system" fn hk_find_next_w(h: HANDLE, data: *mut WIN32_FIND_DATAW) -> i32 {
    if is_merged(h) {
        return match merged_next(h) {
            Some(d) => {
                *data = d;
                1
            }
            None => {
                SetLastError(18); // ERROR_NO_MORE_FILES
                0
            }
        };
    }
    orig!(O_FIND_NEXT_W as FnFindNextW)(h, data)
}

unsafe extern "system" fn hk_find_next_a(h: HANDLE, data: *mut WIN32_FIND_DATAA) -> i32 {
    if is_merged(h) {
        return match merged_next(h) {
            Some(d) => {
                w_to_a(&d, data);
                1
            }
            None => {
                SetLastError(18);
                0
            }
        };
    }
    orig!(O_FIND_NEXT_A as FnFindNextA)(h, data)
}

unsafe extern "system" fn hk_find_close(h: HANDLE) -> i32 {
    let was = MERGED
        .lock()
        .ok()
        .and_then(|mut g| g.as_mut().map(|s| s.remove(&(h as usize))))
        .unwrap_or(false);
    if was {
        drop(Box::from_raw(h as *mut MergedFind));
        return 1;
    }
    orig!(O_FIND_CLOSE as FnFindClose)(h)
}

// ---------------------------------------------------------------------------------------

pub fn install() -> Result<(), String> {
    let kb = "kernelbase.dll";
    let list: &[(&str, usize, &AtomicUsize)] = &[
        ("CreateFileW", hk_create_file_w as usize, &O_CREATE_FILE_W),
        ("CreateFileA", hk_create_file_a as usize, &O_CREATE_FILE_A),
        ("CreateFile2", hk_create_file_2 as usize, &O_CREATE_FILE_2),
        ("GetFileAttributesW", hk_get_attr_w as usize, &O_GET_ATTR_W),
        ("GetFileAttributesA", hk_get_attr_a as usize, &O_GET_ATTR_A),
        ("GetFileAttributesExW", hk_get_attr_ex_w as usize, &O_GET_ATTR_EX_W),
        ("GetFileAttributesExA", hk_get_attr_ex_a as usize, &O_GET_ATTR_EX_A),
        ("SetFileAttributesW", hk_set_attr_w as usize, &O_SET_ATTR_W),
        ("GetDiskFreeSpaceExW", hk_disk_free_ex_w as usize, &O_DISK_FREE_EX_W),
        ("GetDiskFreeSpaceExA", hk_disk_free_ex_a as usize, &O_DISK_FREE_EX_A),
        ("GetDiskFreeSpaceW", hk_disk_free_w as usize, &O_DISK_FREE_W),
        ("GetDiskFreeSpaceA", hk_disk_free_a as usize, &O_DISK_FREE_A),
        ("CreateDirectoryW", hk_create_dir_w as usize, &O_CREATE_DIR_W),
        ("RemoveDirectoryW", hk_remove_dir_w as usize, &O_REMOVE_DIR_W),
        ("DeleteFileW", hk_delete_file_w as usize, &O_DELETE_FILE_W),
        ("MoveFileExW", hk_move_file_ex_w as usize, &O_MOVE_FILE_EX_W),
        ("MoveFileWithProgressW", hk_move_file_with_progress_w as usize, &O_MOVE_FILE_PROGRESS_W),
        ("CopyFileExW", hk_copy_file_ex_w as usize, &O_COPY_FILE_EX_W),
        ("CopyFile2", hk_copy_file_2 as usize, &O_COPY_FILE_2),
        ("FindFirstFileExW", hk_find_first_ex_w as usize, &O_FIND_FIRST_EX_W),
        ("FindFirstFileW", hk_find_first_w as usize, &O_FIND_FIRST_W),
        ("FindFirstFileA", hk_find_first_a as usize, &O_FIND_FIRST_A),
        ("FindNextFileW", hk_find_next_w as usize, &O_FIND_NEXT_W),
        ("FindNextFileA", hk_find_next_a as usize, &O_FIND_NEXT_A),
        ("FindClose", hk_find_close as usize, &O_FIND_CLOSE),
    ];
    let mut errors = Vec::new();
    for (name, det, slot) in list {
        if let Err(e) = unsafe { hook(kb, name, *det, slot) } {
            errors.push(e);
        }
    }
    if errors.is_empty() {
        crate::xlog_info!("VFS: {} file hooks active", list.len());
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}
