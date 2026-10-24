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
                                           at size x size (default 4096), printing timings
  polyflow-cli bake <sculpt.obj> <retopo.obj> <out> [options]
                                           Bake normal maps and ambient occlusion from the sculpt
                                           onto the retopology's UVs with the CPU reference baker,
                                           writing <out>_normal, <out>_object and <out>_ao

Bake options (distances in scene units; defaults scale with the sculpt):
  --size <px>            Map size (default 2048)
  --distance <d>         Ray reach in front of and behind the retopology (default 3% of the
                         sculpt's bounding radius); --frontal and --rear set them apart
  --cage <cage.obj>      Start rays from this cage: the retopology as exported by Polyflow,
                         with vertices moved but none added, removed or reordered
  --cage-push <d>        Start rays from an automatic cage, the retopology pushed out by d
                         along its normals
  --ao-samples <n>       Occlusion rays per texel (default 32)
  --ao-distance <d>      Occluders further away are ignored (default 25% of the radius)
  --directx              Green down (-Y) in the tangent-space normal map, for Unreal
  --cpu                  Bake on the CPU even if a GPU is available (the GPU is used by
                         default; both give the same maps)
  --samples <n>          Antialiasing: n x n samples per texel, 1 to 8 (default 1)
  --padding <px|full>    Extend UV islands this many texels into the background, or over the
                         whole map (default 16)
  --format <f>           png8 (default), png16, exr16 (half float) or exr32 (float)";

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
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
        ["bake", sculpt, retopo, out, options @ ..] => bake(
            PathBuf::from(sculpt),
            PathBuf::from(retopo),
            PathBuf::from(out),
            options,
        ),
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

fn bake(sculpt: PathBuf, retopo: PathBuf, out: PathBuf, options: &[&str]) -> anyhow::Result<()> {
    use polyflow_bake::{BakeControl, BakeSettings, NormalY, Reference};

    let mut size = 2048u32;
    let mut frontal = None;
    let mut rear = None;
    let mut ao_samples = None;
    let mut ao_distance = None;
    let mut directx = false;
    let mut force_cpu = false;
    let mut samples = None;
    // `Some(None)`: fill the whole map.
    let mut padding: Option<Option<u32>> = None;
    let mut format = polyflow_io::image::MapFormat::Png8;
    let mut cage_path = None;
    let mut cage_push = None;
    let mut args = options.iter();
    while let Some(&flag) = args.next() {
        let mut number = |name: &str| -> anyhow::Result<f32> {
            let value = args
                .next()
                .with_context(|| format!("{name} needs a value"))?;
            let value: f32 = value
                .parse()
                .with_context(|| format!("{name} needs a number, got `{value}`"))?;
            if !(value >= 0.0 && value.is_finite()) {
                bail!("{name} must be a positive number");
            }
            Ok(value)
        };
        match flag {
            "--size" => size = number(flag)? as u32,
            "--distance" => {
                let d = number(flag)?;
                (frontal, rear) = (Some(d), Some(d));
            }
            "--frontal" => frontal = Some(number(flag)?),
            "--rear" => rear = Some(number(flag)?),
            "--ao-samples" => ao_samples = Some(number(flag)? as u32),
            "--ao-distance" => ao_distance = Some(number(flag)?),
            "--directx" => directx = true,
            "--cpu" => force_cpu = true,
            "--samples" => {
                let n = number(flag)? as u32;
                if !(1..=polyflow_bake::MAX_SAMPLES).contains(&n) {
                    bail!(
                        "--samples must be between 1 and {}",
                        polyflow_bake::MAX_SAMPLES
                    );
                }
                samples = Some(n);
            }
            "--padding" => {
                let value = args.next().context("--padding needs a value")?;
                padding = Some(match *value {
                    "full" => None,
                    px => Some(px.parse().with_context(|| {
                        format!("--padding needs a number of texels or `full`, got `{px}`")
                    })?),
                });
            }
            "--format" => {
                let value = args.next().context("--format needs a value")?;
                format = polyflow_io::image::MapFormat::from_name(value).with_context(|| {
                    format!("unknown format `{value}`: use png8, png16, exr16 or exr32")
                })?;
            }
            "--cage" => {
                let path = args.next().context("--cage needs a file")?;
                cage_path = Some(PathBuf::from(path));
            }
            "--cage-push" => cage_push = Some(number(flag)?),
            other => bail!("unknown bake option `{other}`"),
        }
    }
    if !(1..=polyflow_bake::raster::MAX_SIZE).contains(&size) {
        bail!(
            "size must be between 1 and {}",
            polyflow_bake::raster::MAX_SIZE
        );
    }

    let start = Instant::now();
    let loaded =
        polyflow_io::load_mesh(&sculpt).with_context(|| format!("loading {}", sculpt.display()))?;
    let high = loaded.mesh;
    let bvh = polyflow_spatial::Bvh::build(&high);
    println!(
        "sculpt      {} triangles, loaded with its BVH in {:.2} s",
        high.triangle_count(),
        start.elapsed().as_secs_f64()
    );

    let start = Instant::now();
    let imported = polyflow_io::retopo::load(&retopo)
        .with_context(|| format!("loading {}", retopo.display()))?;
    if cage_path.is_some() && cage_push.is_some() {
        bail!("use either --cage or --cage-push, not both");
    }
    let cage = match (&cage_path, cage_push) {
        (Some(path), _) => {
            let cage = polyflow_io::retopo::load(path)
                .with_context(|| format!("loading {}", path.display()))?;
            let cage = polyflow_bake::Cage::from_mesh(&imported.mesh, &cage.mesh)
                .with_context(|| format!("{} does not fit the retopology", path.display()))?;
            Some(cage)
        }
        (None, Some(push)) => Some(polyflow_bake::Cage::automatic(&imported.mesh, push)),
        (None, None) => None,
    };
    let low = polyflow_bake::BakeMesh::with_cage(&imported.mesh, cage.as_ref());
    println!(
        "retopology  {} triangles, prepared in {:.2} s",
        low.triangle_count(),
        start.elapsed().as_secs_f64()
    );
    if low.faces_without_uvs > 0 {
        println!("no UVs      {} faces left out", low.faces_without_uvs);
    }
    if low.triangle_count() == 0 {
        bail!("the retopology has no faces with UVs to bake onto");
    }

    let radius = high.bounds().radius();
    let mut settings = BakeSettings::for_scene(size, size, radius);
    settings.frontal_distance = frontal.unwrap_or(settings.frontal_distance);
    settings.rear_distance = rear.unwrap_or(settings.rear_distance);
    settings.ao_samples = ao_samples.unwrap_or(settings.ao_samples);
    settings.ao_distance = ao_distance.unwrap_or(settings.ao_distance);
    if directx {
        settings.normal_y = NormalY::DirectX;
    }
    settings.samples = samples.unwrap_or(settings.samples);
    if let Some(padding) = padding {
        settings.padding = padding;
    }
    let start_from = match (&cage_path, cage_push) {
        (Some(path), _) => format!("from the cage {}", path.display()),
        (None, Some(push)) => format!("from an automatic cage {push} out"),
        (None, None) => format!("{} in front", settings.frontal_distance),
    };
    println!(
        "settings    {size}x{size}, rays {start_from} / {} behind, AO {} rays within {}",
        settings.rear_distance, settings.ao_samples, settings.ao_distance
    );
    let padding_text = settings
        .padding
        .map_or("full".to_owned(), |p| format!("{p} texels"));
    println!(
        "            {n}x{n} samples per texel, padding {padding_text}, {}",
        format.name(),
        n = settings.samples
    );

    let reference = Reference {
        mesh: &high,
        bvh: &bvh,
    };
    let gpu = if force_cpu {
        None
    } else {
        match polyflow_render::context::GpuContext::headless() {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                println!("device      no usable GPU ({e}); baking on the CPU");
                None
            }
        }
    };
    let start = Instant::now();
    let baked = match &gpu {
        Some(ctx) => {
            let tracer = match polyflow_render::bake::Tracer::Auto.resolve(ctx) {
                polyflow_render::bake::Tracer::RayQuery => "hardware ray tracing",
                _ => "compute ray tracing",
            };
            println!("device      {}, {tracer}", ctx.device_name());
            polyflow_render::bake::bake(ctx, &low, reference, &settings, &BakeControl::default())?
        }
        None => {
            println!("device      CPU, {} threads", rayon::current_num_threads());
            polyflow_bake::bake(&low, reference, &settings, &BakeControl::default())?
        }
    };
    let time = start.elapsed().as_secs_f64();
    // Per sample: the projection ray, plus occlusion rays when it hits.
    let samples = (settings.samples * settings.samples) as f64;
    let rays =
        samples * (baked.hits as f64 * (1.0 + settings.ao_samples as f64) + baked.misses as f64);
    println!(
        "bake        {time:.2} s ({:.1} M rays/s)",
        rays / time.max(1e-9) / 1e6
    );
    let covered = baked.hits + baked.misses;
    println!(
        "texels      {} covered, {} missed ({:.2}%)",
        covered,
        baked.misses,
        100.0 * baked.misses as f64 / covered.max(1) as f64
    );
    if baked.misses * 100 > covered {
        println!(
            "            more than 1% missed: try a larger --distance (or --frontal / --rear)"
        );
    }

    let maps = [
        ("normal", &baked.tangent_normal),
        ("object", &baked.object_normal),
        ("ao", &baked.ambient_occlusion),
    ];
    for (suffix, map) in maps {
        let Some(map) = map else { continue };
        let mut name = out.file_name().unwrap_or_default().to_owned();
        name.push(format!("_{suffix}.{}", format.extension()));
        let path = out.with_file_name(name);
        polyflow_io::image::save_map(
            &path,
            format,
            map.width,
            map.height,
            map.channels,
            &map.data,
        )
        .with_context(|| format!("writing {}", path.display()))?;
        println!("wrote       {}", path.display());
    }
    Ok(())
}
