//! Inline hooks on real Windows functions (via `retour`), used where calls into Windows
//! do not go through the game's import table: file APIs called by the CRT, exception
//! machinery, etc.

use std::sync::atomic::{AtomicUsize, Ordering};

use retour::RawDetour;

use crate::util::wide;

/// Detour `module!func` to `detour`; the trampoline (call-original) address is stored
/// in `orig`. The detour object is leaked on purpose – hooks live for the process.
pub unsafe fn hook(module: &str, func: &str, detour: usize, orig: &AtomicUsize) -> Result<(), String> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
    let mut h = GetModuleHandleW(wide(module).as_ptr());
    if h.is_null() {
        h = LoadLibraryW(wide(module).as_ptr());
    }
    if h.is_null() {
        return Err(format!("hook {module}!{func}: module not found"));
    }
    let cname = std::ffi::CString::new(func).unwrap();
    let target = GetProcAddress(h, cname.as_ptr() as *const u8).ok_or_else(|| format!("hook: {module}!{func} not found"))?;
    let d = RawDetour::new(target as *const (), detour as *const ()).map_err(|e| format!("hook {module}!{func}: {e}"))?;
    orig.store(d.trampoline() as *const () as usize, Ordering::Release);
    d.enable().map_err(|e| format!("hook {module}!{func}: {e}"))?;
    Box::leak(Box::new(d));
    crate::xlog_debug!("hook {module}!{func} active");
    Ok(())
}

#[cfg(test)]
mod movefile_probe {
    use super::*;
    use std::sync::atomic::AtomicU32;

    static EX: AtomicU32 = AtomicU32::new(0);
    static PROG: AtomicU32 = AtomicU32::new(0);
    static O_EX: AtomicUsize = AtomicUsize::new(0);
    static O_PROG: AtomicUsize = AtomicUsize::new(0);
    type Ex = unsafe extern "system" fn(*const u16, *const u16, u32) -> i32;
    type Prog = unsafe extern "system" fn(*const u16, *const u16, usize, usize, u32) -> i32;
    unsafe extern "system" fn ex(a: *const u16, b: *const u16, f: u32) -> i32 {
        EX.fetch_add(1, Ordering::SeqCst);
        std::mem::transmute::<usize, Ex>(O_EX.load(Ordering::SeqCst))(a, b, f)
    }
    unsafe extern "system" fn prog(a: *const u16, b: *const u16, c: usize, d: usize, f: u32) -> i32 {
        PROG.fetch_add(1, Ordering::SeqCst);
        std::mem::transmute::<usize, Prog>(O_PROG.load(Ordering::SeqCst))(a, b, c, d, f)
    }

    /// Which kernelbase entry each move API reaches (run with --nocapture).
    #[test]
    #[ignore]
    fn movefile_probe() {
        use windows_sys::Win32::Storage::FileSystem::{MoveFileA, MoveFileExA, MoveFileExW, MoveFileW, MoveFileWithProgressW};
        unsafe {
            hook("kernelbase.dll", "MoveFileExW", ex as usize, &O_EX).unwrap();
            hook("kernelbase.dll", "MoveFileWithProgressW", prog as usize, &O_PROG).unwrap();
            let dir = std::env::temp_dir().join(format!("xcl-movefile-probe-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&dir);
            let (a, b) = (dir.join("a"), dir.join("b"));
            let mut report = Vec::new();
            for name in ["MoveFileW", "MoveFileA", "MoveFileExW", "MoveFileExA", "MoveFileWithProgressW"] {
                std::fs::write(&a, b"x").unwrap();
                let _ = std::fs::remove_file(&b);
                let (e0, p0) = (EX.load(Ordering::SeqCst), PROG.load(Ordering::SeqCst));
                let (wa, wb) = (wide(&a.to_string_lossy()), wide(&b.to_string_lossy()));
                let na = std::ffi::CString::new(a.to_string_lossy().as_bytes()).unwrap();
                let nb = std::ffi::CString::new(b.to_string_lossy().as_bytes()).unwrap();
                let ok = match name {
                    "MoveFileW" => MoveFileW(wa.as_ptr(), wb.as_ptr()),
                    "MoveFileA" => MoveFileA(na.as_ptr() as _, nb.as_ptr() as _),
                    "MoveFileExW" => MoveFileExW(wa.as_ptr(), wb.as_ptr(), 0),
                    "MoveFileExA" => MoveFileExA(na.as_ptr() as _, nb.as_ptr() as _, 0),
                    _ => MoveFileWithProgressW(wa.as_ptr(), wb.as_ptr(), None, std::ptr::null(), 0),
                };
                report.push(format!(
                    "{name}: ok={ok} reaches MoveFileExW={} MoveFileWithProgressW={}",
                    EX.load(Ordering::SeqCst) - e0,
                    PROG.load(Ordering::SeqCst) - p0
                ));
            }
            let _ = std::fs::remove_dir_all(&dir);
            for l in &report {
                println!("PROBE {l}");
            }
        }
    }
}
