#![allow(
    dead_code,
    reason = "each test binary compiles this file and uses a different subset of it"
)]

use std::path::PathBuf;
use std::process::Command;

pub const VULKAN_1_0: &str = "vulkan1.0";
pub const VULKAN_1_1: &str = "vulkan1.1";

pub fn validator() -> Option<PathBuf> {
    if let Some(from_env) = std::env::var_os("SPIRV_VAL") {
        let path = PathBuf::from(from_env);
        assert!(
            path.is_file(),
            "SPIRV_VAL points at {path:?} and there is no file there — every validation test would \
             skip over that, and skips are invisible. Unset it to look in the usual place."
        );
        return Some(path);
    }

    let name = if cfg!(windows) {
        "spirv-val.exe"
    } else {
        "spirv-val"
    };
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|directory| directory.join(name))
        .find(|candidate| candidate.is_file())
}

pub fn validate(words: &[u32], label: &str, target_env: &str) -> Result<(), String> {
    let Some(tool) = validator() else {
        eprintln!("SKIPPED {label}: spirv-val not found (set SPIRV_VAL)");
        return Ok(());
    };

    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }

    let stem: String = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '_'
            }
        })
        .collect();

    let path = std::env::temp_dir().join(format!("simdr-{stem}.spv"));
    std::fs::write(&path, &bytes).expect("the temp directory is writable");

    let output = Command::new(&tool)
        .arg("--target-env")
        .arg(target_env)
        .arg(&path)
        .output()
        .expect("spirv-val is executable");

    if output.status.success() {
        return Ok(());
    }

    Err(format!(
        "spirv-val rejected {label}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    ))
}

pub fn expect_valid(words: &[u32], label: &str, target_env: &str) {
    if let Err(complaint) = validate(words, label, target_env) {
        panic!("{complaint}");
    }
}
