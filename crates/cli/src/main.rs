//! Headless command-line interface for batch baking and conversion.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context as _, bail};

const USAGE: &str = "\
Usage:
  polyflow-cli info <mesh.obj>             Load a mesh and print statistics and timings
  polyflow-cli torus <rings> <out.obj>     Write a test torus with 2*rings*(rings/2) triangles";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["info", path] => info(PathBuf::from(path)),
        ["torus", rings, out] => torus(rings, PathBuf::from(out)),
        ["--version" | "-V"] => {
            println!("polyflow-cli {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => {
            eprintln!("polyflow-cli {}\n\n{USAGE}", env!("CARGO_PKG_VERSION"));
            return ExitCode::FAILURE;
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn info(path: PathBuf) -> anyhow::Result<()> {
    let start = Instant::now();
    let loaded =
        polyflow_io::load_mesh(&path).with_context(|| format!("loading {}", path.display()))?;
    let total = start.elapsed();
    let mesh = &loaded.mesh;
    let bounds = mesh.bounds();
    let mb = |bytes: f64| bytes / (1024.0 * 1024.0);
    let secs = |d: std::time::Duration| d.as_secs_f64();

    println!("file        {}", path.display());
    println!("size        {:.1} MB", mb(loaded.file_bytes as f64));
    println!("vertices    {}", mesh.vertex_count());
    println!("triangles   {}", mesh.triangle_count());
    if loaded.skipped_faces > 0 {
        println!(
            "skipped     {} faces with fewer than 3 vertices",
            loaded.skipped_faces
        );
    }
    println!("bounds      {} .. {}", bounds.min, bounds.max);
    println!("read        {:.3} s", secs(loaded.read_time));
    println!(
        "parse       {:.3} s ({:.0} MB/s)",
        secs(loaded.parse_time),
        mb(loaded.file_bytes as f64) / secs(loaded.parse_time).max(1e-9)
    );
    println!("normals     {:.3} s", secs(loaded.normals_time));
    println!("total       {:.3} s", secs(total));
    Ok(())
}

fn torus(rings: &str, out: PathBuf) -> anyhow::Result<()> {
    let rings: u32 = rings.parse().context("rings must be a positive integer")?;
    if rings < 3 {
        bail!("rings must be at least 3");
    }
    let mesh = polyflow_core::primitives::torus(1.0, 0.35, rings, (rings / 2).max(3));
    let file =
        std::fs::File::create(&out).with_context(|| format!("creating {}", out.display()))?;
    polyflow_io::obj::write(&mesh, file)?;
    println!(
        "wrote {} triangles to {}",
        mesh.triangle_count(),
        out.display()
    );
    Ok(())
}
