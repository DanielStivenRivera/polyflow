//! Compiles the Slang shaders in `shaders/` to SPIR-V.
//!
//! `slangc` is looked up in `$SLANGC`, then `$VULKAN_SDK/Bin`, then `PATH`.
//! Precompiled SPIR-V is committed in `shaders/spirv/` so the crate also builds
//! without the Vulkan SDK (CI, casual contributors). `shaders/spirv/sources.hash`
//! records which shader sources it was built from: when the sources change and
//! `slangc` is available, the committed copy is refreshed so it can be committed
//! alongside the source change; without `slangc` a stale copy is a build error.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

/// Shaders compiled to SPIR-V; other `.slang` files are only imported.
const SHADERS: &[&str] = &["bake", "bake_rt", "cull", "mesh", "overlay"];

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let src_dir = manifest_dir.join("shaders");
    let spirv_dir = src_dir.join("spirv");
    let hash_file = spirv_dir.join("sources.hash");

    println!("cargo:rerun-if-env-changed=SLANGC");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");
    println!("cargo:rerun-if-changed=shaders");

    let hash = format!("{:016x}\n", sources_hash(&src_dir));
    let up_to_date = fs::read_to_string(&hash_file).is_ok_and(|h| h == hash);

    match find_slangc() {
        Some(slangc) => {
            for name in SHADERS {
                compile(
                    &slangc,
                    &src_dir.join(format!("{name}.slang")),
                    &out_dir.join(format!("{name}.spv")),
                );
            }
            if !up_to_date {
                fs::create_dir_all(&spirv_dir).unwrap();
                for name in SHADERS {
                    let file = format!("{name}.spv");
                    fs::copy(out_dir.join(&file), spirv_dir.join(&file)).unwrap();
                }
                fs::write(&hash_file, &hash).unwrap();
                println!(
                    "cargo:warning=refreshed precompiled SPIR-V in shaders/spirv/; commit it with your shader changes"
                );
            }
        }
        None => {
            if !up_to_date {
                panic!(
                    "shaders/spirv/ is out of date with the Slang sources and `slangc` was not found. \
                     Install the Vulkan SDK (https://vulkan.lunarg.com/) or set $SLANGC."
                );
            }
            for name in SHADERS {
                let file = format!("{name}.spv");
                fs::copy(spirv_dir.join(&file), out_dir.join(&file)).unwrap();
            }
        }
    }
}

fn find_slangc() -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        "slangc.exe"
    } else {
        "slangc"
    };
    let candidates = env::var_os("SLANGC")
        .map(PathBuf::from)
        .into_iter()
        .chain(env::var_os("VULKAN_SDK").map(|sdk| Path::new(&sdk).join("Bin").join(exe)))
        .chain(env::var_os("VULKAN_SDK").map(|sdk| Path::new(&sdk).join("bin").join(exe)))
        .chain(Some(PathBuf::from(exe)));
    candidates.into_iter().find(|c| {
        Command::new(c)
            .arg("-v")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

fn compile(slangc: &Path, src: &Path, dst: &Path) {
    let output = Command::new(slangc)
        .arg(src)
        .arg("-I")
        .arg(src.parent().unwrap())
        .args(["-target", "spirv", "-profile", "spirv_1_6"])
        // glam matrices are column-major.
        .arg("-matrix-layout-column-major")
        .arg("-fvk-use-entrypoint-name")
        .arg("-o")
        .arg(dst)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {}: {e}", slangc.display()));
    if !output.status.success() {
        panic!(
            "slangc failed on {}:\n{}{}",
            src.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

/// FNV-1a over every `.slang` file (name and contents, line endings
/// normalized so Windows and Linux checkouts agree).
fn sources_hash(dir: &Path) -> u64 {
    let mut files: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "slang"))
        .collect();
    files.sort();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    };
    for file in files {
        feed(file.file_name().unwrap().as_encoded_bytes());
        let text = fs::read_to_string(&file).unwrap().replace("\r\n", "\n");
        feed(text.as_bytes());
    }
    hash
}
