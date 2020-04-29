
use std::path::Path;
use std::process::Command;

#[cfg(target_os = "macos")]
const SYSTEM_LINKER: &str = "ld";
#[cfg(target_os = "linux")]
const SYSTEM_LINKER: &str = "ld";
#[cfg(target_os = "windows")]
const SYSTEM_LINKER: &str = "link.exe";

pub fn link(path: impl AsRef<Path>) {
    let status = Command::new(SYSTEM_LINKER)
        .arg(path.as_ref())
        .arg("-lc")
        .status()
        .unwrap();

    if !status.success() {
        panic!("failed to link: {}", status.code().unwrap_or(1));
    }
}

