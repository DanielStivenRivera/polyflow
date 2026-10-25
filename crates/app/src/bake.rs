//! Baking from the app: on a background thread, with progress and cancel.
//!
//! The thread creates its own headless GPU context rather than sharing the
//! renderer's, so the viewport keeps drawing (Vulkan queues may not be used
//! from two threads at once) and a GPU failure cannot take the window down.
//! Without a usable GPU it bakes on the CPU. Maps and the low-poly as baked
//! are written next to each other, then the result comes back to the event
//! loop for the panel and the preview.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use polyflow_bake::{BakeControl, BakeMesh, BakeSettings, Baked, MapSet, NormalY, Reference};
use polyflow_core::mesh::TriMesh;
use polyflow_io::image::MapFormat;
use polyflow_io::project::BakeState;
use polyflow_spatial::Bvh;
use winit::event_loop::EventLoopProxy;

use crate::UserEvent;

/// Map sizes offered in the panel.
pub const SIZES: [u32; 5] = [512, 1024, 2048, 4096, 8192];

/// A bake in progress.
#[derive(Debug)]
pub struct BakeJob {
    pub control: Arc<BakeControl>,
    pub total_rows: u32,
    pub started: Instant,
}

impl BakeJob {
    /// How far along the bake is, 0 to 1.
    pub fn progress(&self) -> f32 {
        self.control.rows_done() as f32 / self.total_rows.max(1) as f32
    }
}

/// A finished bake.
#[derive(Debug)]
pub struct BakeReport {
    pub files: Vec<PathBuf>,
    pub time: Duration,
    /// What baked it, e.g. "NVIDIA GeForce RTX 4070 (hardware ray tracing)".
    pub device: String,
    pub baked: Baked,
    /// The low-poly as baked, for the preview.
    pub low: BakeMesh,
    /// Green down in the tangent-space normal map.
    pub directx: bool,
}

#[derive(Debug)]
pub enum BakeOutcome {
    Done(Box<BakeReport>),
    Cancelled,
    Failed(String),
}

/// The baker's settings from the panel's, for a sculpt of bounding-sphere
/// `radius`.
pub fn settings(state: &BakeState, radius: f32) -> BakeSettings {
    let defaults = BakeSettings::for_scene(state.size, state.size, radius);
    BakeSettings {
        frontal_distance: state.frontal.unwrap_or(defaults.frontal_distance),
        rear_distance: state.rear.unwrap_or(defaults.rear_distance),
        maps: MapSet {
            tangent_normal: state.tangent_normal,
            object_normal: state.object_normal,
            ambient_occlusion: state.ambient_occlusion,
        },
        normal_y: if state.directx {
            NormalY::DirectX
        } else {
            NormalY::OpenGl
        },
        ao_samples: state.ao_samples.max(1),
        ao_distance: state.ao_distance.unwrap_or(defaults.ao_distance),
        samples: state.samples.clamp(1, polyflow_bake::MAX_SAMPLES),
        padding: (!state.pad_full).then_some(state.padding),
        ..defaults
    }
}

/// Where the maps go: `<output>_normal.png` and so on. Defaults to the
/// project's name next to it, else the sculpt's.
pub fn output_prefix(state: &BakeState, project: Option<&Path>, sculpt: Option<&Path>) -> PathBuf {
    if let Some(output) = &state.output {
        return PathBuf::from(output);
    }
    match project.or(sculpt) {
        Some(path) => path.with_extension(""),
        None => PathBuf::from("bake"),
    }
}

/// `prefix` + `_suffix.extension`.
pub fn output_file(prefix: &Path, suffix: &str, extension: &str) -> PathBuf {
    let mut name = prefix.file_name().unwrap_or_default().to_owned();
    name.push(format!("_{suffix}.{extension}"));
    prefix.with_file_name(name)
}

/// Starts baking on a background thread; the outcome arrives as
/// [`UserEvent::BakeDone`].
#[allow(clippy::too_many_arguments)]
pub fn start(
    proxy: EventLoopProxy<UserEvent>,
    low: BakeMesh,
    mesh: Arc<TriMesh>,
    bvh: Arc<Bvh>,
    settings: BakeSettings,
    format: MapFormat,
    prefix: PathBuf,
    cpu: bool,
) -> std::io::Result<BakeJob> {
    let control = Arc::new(BakeControl::default());
    let job = BakeJob {
        control: Arc::clone(&control),
        total_rows: settings.progress_rows(),
        started: Instant::now(),
    };
    std::thread::Builder::new()
        .name("bake".into())
        .spawn(move || {
            let outcome = run(&low, &mesh, &bvh, &settings, format, &prefix, cpu, &control);
            let directx = settings.normal_y == NormalY::DirectX;
            let outcome = match outcome {
                Ok(Some((files, time, device, baked))) => BakeOutcome::Done(Box::new(BakeReport {
                    files,
                    time,
                    device,
                    baked,
                    low,
                    directx,
                })),
                Ok(None) => BakeOutcome::Cancelled,
                Err(e) => BakeOutcome::Failed(e),
            };
            // Fails only if the event loop is gone, i.e. the app is closing.
            let _ = proxy.send_event(UserEvent::BakeDone(Box::new(outcome)));
        })?;
    Ok(job)
}

type Finished = (Vec<PathBuf>, Duration, String, Baked);

/// Bakes and saves; `Ok(None)` if cancelled.
#[allow(clippy::too_many_arguments)]
fn run(
    low: &BakeMesh,
    mesh: &TriMesh,
    bvh: &Bvh,
    settings: &BakeSettings,
    format: MapFormat,
    prefix: &Path,
    cpu: bool,
    control: &BakeControl,
) -> Result<Option<Finished>, String> {
    let start = Instant::now();
    let reference = Reference { mesh, bvh };
    let gpu = if cpu {
        None
    } else {
        polyflow_render::context::GpuContext::headless()
            .inspect_err(|e| log::warn!("No GPU for baking ({e}); baking on the CPU"))
            .ok()
    };
    let (baked, device) = match &gpu {
        Some(ctx) => {
            let tracer = match polyflow_render::bake::Tracer::Auto.resolve(ctx) {
                polyflow_render::bake::Tracer::RayQuery => "hardware ray tracing",
                _ => "compute",
            };
            let device = format!("{} ({tracer})", ctx.device_name());
            match polyflow_render::bake::bake(ctx, low, reference, settings, control) {
                Ok(baked) => (baked, device),
                Err(polyflow_render::bake::GpuBakeError::Cancelled) => return Ok(None),
                Err(e) => return Err(format!("The GPU bake failed: {e}")),
            }
        }
        None => match polyflow_bake::bake(low, reference, settings, control) {
            Ok(baked) => (
                baked,
                format!("CPU ({} threads)", rayon::current_num_threads()),
            ),
            Err(_) => return Ok(None),
        },
    };
    drop(gpu);
    let time = start.elapsed();

    if let Some(dir) = prefix.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("Could not create {}: {e}", dir.display()))?;
    }
    let mut files = Vec::new();
    for (suffix, map) in [
        ("normal", &baked.tangent_normal),
        ("object", &baked.object_normal),
        ("ao", &baked.ambient_occlusion),
    ] {
        let Some(map) = map else { continue };
        let path = output_file(prefix, suffix, format.extension());
        polyflow_io::image::save_map(
            &path,
            format,
            map.width,
            map.height,
            map.channels,
            &map.data,
        )
        .map_err(|e| format!("Could not write {}: {e}", path.display()))?;
        files.push(path);
    }
    let path = output_file(prefix, "low", "obj");
    let corners: Vec<polyflow_io::lowpoly::MeshCorner> = low
        .corners
        .iter()
        .map(|c| polyflow_io::lowpoly::MeshCorner {
            position: c.position,
            normal: c.normal,
            uv: c.uv,
            tangent: c.tangent,
        })
        .collect();
    polyflow_io::lowpoly::save_obj(&path, &corners)
        .map_err(|e| format!("Could not write {}: {e}", path.display()))?;
    files.push(path);
    let map_file = |suffix: &str| {
        let path = output_file(prefix, suffix, format.extension());
        files.contains(&path).then_some(path)
    };
    let directx = settings.normal_y == NormalY::DirectX;
    let normal = map_file("normal").filter(|_| !directx);
    let textures =
        polyflow_io::gltf::BakedTextures::from_files(normal.as_deref(), map_file("ao").as_deref());
    let path = output_file(prefix, "low", "glb");
    polyflow_io::gltf::save_glb(&path, &corners, &textures)
        .map_err(|e| format!("Could not write {}: {e}", path.display()))?;
    files.push(path);
    Ok(Some((files, time, device, baked)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_names() {
        let state = BakeState::default();
        let project = Path::new("work/hero.pflow");
        assert_eq!(
            output_prefix(&state, Some(project), None),
            Path::new("work/hero")
        );
        assert_eq!(
            output_prefix(&state, None, Some(Path::new("s/sculpt.obj"))),
            Path::new("s/sculpt")
        );
        let custom = BakeState {
            output: Some("maps/final".into()),
            ..BakeState::default()
        };
        assert_eq!(
            output_prefix(&custom, Some(project), None),
            Path::new("maps/final")
        );
        assert_eq!(
            output_file(Path::new("maps/final"), "ao", "exr"),
            Path::new("maps/final_ao.exr")
        );
    }

    #[test]
    fn settings_follow_the_panel() {
        let state = BakeState {
            size: 1024,
            samples: 2,
            object_normal: true,
            directx: true,
            pad_full: true,
            rear: Some(0.5),
            ..BakeState::default()
        };
        let s = settings(&state, 10.0);
        assert_eq!((s.width, s.height, s.samples), (1024, 1024, 2));
        assert!(s.maps.tangent_normal && s.maps.object_normal && s.maps.ambient_occlusion);
        assert_eq!(s.normal_y, NormalY::DirectX);
        assert_eq!(s.padding, None);
        assert_eq!(s.rear_distance, 0.5);
        assert!(
            (s.frontal_distance - 0.3).abs() < 1e-6,
            "3% of the radius by default"
        );
    }
}
