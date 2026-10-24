//! Headless command-line interface for batch baking and conversion.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context as _, bail};

const USAGE: &str = "\
Usage:
  polyflow-cli info <mesh.obj>             Load a mesh and print statistics and timings
  polyflow-cli bvh <mesh.obj>              Build the BVH and time raycasts and closest-point queries
  polyflow-cli torus <rings> <out.obj>     Write a test torus with 2*rings*(rings/2) triangles
  polyflow-cli raster <retopo.obj> [size]  Prepare a retopology for baking and rasterize its UVs
                                           at size x size (default 4096), printing timings";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["info", path] => info(PathBuf::from(path)),
        ["bvh", path] => bvh(PathBuf::from(path)),
        ["torus", rings, out] => torus(rings, PathBuf::from(out)),
        ["raster", path] => raster(PathBuf::from(path), "4096"),
        ["raster", path, size] => raster(PathBuf::from(path), size),
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

fn bvh(path: PathBuf) -> anyhow::Result<()> {
    use polyflow_core::glam::Vec3;
    use rayon::prelude::*;

    let loaded =
        polyflow_io::load_mesh(&path).with_context(|| format!("loading {}", path.display()))?;
    let mesh = &loaded.mesh;
    println!("triangles   {}", mesh.triangle_count());

    let start = Instant::now();
    let bvh = polyflow_spatial::Bvh::build(mesh);
    let build = start.elapsed();
    println!("build       {:.3} s", build.as_secs_f64());
    println!("nodes       {}", bvh.node_count());
    println!(
        "memory      {:.1} MB ({:.1} bytes/triangle)",
        bvh.memory_bytes() as f64 / (1024.0 * 1024.0),
        bvh.memory_bytes() as f64 / mesh.triangle_count() as f64
    );

    // Queries like the viewport's: rays from around the mesh towards random
    // surface points, and points scattered near the surface.
    const QUERIES: usize = 200_000;
    let bounds = mesh.bounds();
    let radius = bounds.radius();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut random = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut rays = Vec::with_capacity(QUERIES);
    let mut points = Vec::with_capacity(QUERIES);
    for _ in 0..QUERIES {
        let tri = (random() * mesh.triangle_count() as f32) as usize % mesh.triangle_count();
        let target = mesh.positions[mesh.indices[tri * 3] as usize];
        let dir = Vec3::new(random() - 0.5, random() - 0.5, random() - 0.5).normalize_or(Vec3::X);
        rays.push((target + dir * radius * 2.0, -dir));
        points.push(target + dir * radius * 0.01 * random());
    }

    let time = |name: &str, f: &(dyn Fn(usize) -> bool + Sync)| {
        let start = Instant::now();
        let hits = (0..QUERIES).filter(|&i| f(i)).count();
        let single = start.elapsed().as_secs_f64() / QUERIES as f64;
        let start = Instant::now();
        let _ = (0..QUERIES).into_par_iter().filter(|&i| f(i)).count();
        let parallel = QUERIES as f64 / start.elapsed().as_secs_f64();
        println!(
            "{name:<11} {:.2} us/query on one thread, {:.1} M queries/s on all cores ({:.0}% hit)",
            single * 1e6,
            parallel / 1e6,
            100.0 * hits as f64 / QUERIES as f64
        );
    };
    time("raycast", &|i| {
        bvh.raycast(mesh, rays[i].0, rays[i].1, f32::INFINITY)
            .is_some()
    });
    time("closest", &|i| {
        bvh.closest_point(mesh, points[i], f32::INFINITY).is_some()
    });
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

fn raster(path: PathBuf, size: &str) -> anyhow::Result<()> {
    let size: u32 = size.parse().context("size must be a positive integer")?;
    if !(1..=polyflow_bake::raster::MAX_SIZE).contains(&size) {
        bail!(
            "size must be between 1 and {}",
            polyflow_bake::raster::MAX_SIZE
        );
    }
    let start = Instant::now();
    let imported =
        polyflow_io::retopo::load(&path).with_context(|| format!("loading {}", path.display()))?;
    let load = start.elapsed();
    let retopo = &imported.mesh;
    println!(
        "retopology  {} faces, {} vertices",
        retopo.face_count(),
        retopo.vertex_count()
    );
    println!("load        {:.3} s", load.as_secs_f64());

    let start = Instant::now();
    let mesh = polyflow_bake::BakeMesh::from_retopo(retopo);
    println!(
        "bake mesh   {:.3} s: {} triangles, normals and MikkTSpace tangents",
        start.elapsed().as_secs_f64(),
        mesh.triangle_count()
    );
    if mesh.faces_without_uvs > 0 {
        println!("no UVs      {} faces left out", mesh.faces_without_uvs);
    }

    let start = Instant::now();
    let raster = polyflow_bake::rasterize(&mesh, size, size, polyflow_core::glam::Vec2::splat(0.5));
    let time = start.elapsed().as_secs_f64();
    println!(
        "rasterize   {time:.3} s at {size}x{size} ({:.0} M texels/s)",
        (size as f64 * size as f64) / time / 1e6
    );
    println!("coverage    {:.1}%", raster.coverage() * 100.0);
    if raster.overlaps > 0 {
        println!(
            "overlaps    {} texels covered twice or more",
            raster.overlaps
        );
    }
    if raster.degenerate > 0 {
        println!(
            "degenerate  {} triangles with zero UV area",
            raster.degenerate
        );
    }
    if raster.out_of_range > 0 {
        println!("far UVs     {} triangles ignored", raster.out_of_range);
    }
    Ok(())
}
