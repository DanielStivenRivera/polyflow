//! Background preparation of reference meshes: loading or generating the
//! mesh and building its GPU layout, which can take seconds for huge meshes.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use polyflow_core::mesh::TriMesh;
use polyflow_core::primitives;
use polyflow_render::MeshData;
use polyflow_spatial::Bvh;

use crate::ui::{FileInfo, MeshInfo, MeshSource};

#[derive(Debug, Clone)]
pub enum Job {
    Open(PathBuf),
    Torus { rings: u32 },
}

impl Job {
    /// Short description for the UI.
    pub fn name(&self) -> String {
        match self {
            Job::Open(path) => display_name(path),
            Job::Torus { rings } => format!("test torus ({rings} rings)"),
        }
    }
}

/// A mesh ready to upload. `info.upload` is filled in after uploading.
#[derive(Debug)]
pub struct Prepared {
    pub mesh: TriMesh,
    pub gpu_data: MeshData,
    pub bvh: Bvh,
    pub info: MeshInfo,
}

pub fn run(job: &Job) -> Result<Prepared, String> {
    let (mesh, name, source) = match job {
        Job::Open(path) => {
            let loaded = polyflow_io::load_mesh(path)
                .map_err(|e| format!("Could not open {}: {e}", job.name()))?;
            let file = FileInfo {
                bytes: loaded.file_bytes,
                skipped_faces: loaded.skipped_faces,
                read: loaded.read_time,
                parse: loaded.parse_time,
                normals: loaded.normals_time,
            };
            (loaded.mesh, display_name(path), MeshSource::File(file))
        }
        Job::Torus { rings } => {
            let start = Instant::now();
            let mesh = primitives::torus(1.0, 0.35, *rings, torus_sides(*rings));
            (
                mesh,
                "Test torus".into(),
                MeshSource::Generated {
                    time: start.elapsed(),
                },
            )
        }
    };
    // Both use all cores; running them together keeps the cores busy through
    // each one's serial phases.
    let ((gpu_data, bvh), bvh_build) = {
        let mut bvh_time = Duration::ZERO;
        let result = rayon::join(
            || MeshData::build(&mesh),
            || {
                let start = Instant::now();
                let bvh = Bvh::build(&mesh);
                bvh_time = start.elapsed();
                bvh
            },
        );
        (result, bvh_time)
    };
    let info = MeshInfo {
        name,
        vertices: mesh.vertex_count(),
        triangles: mesh.triangle_count(),
        meshlets: gpu_data.meshlet_count(),
        source,
        meshlet_build: gpu_data.build_time,
        bvh_build,
        bvh_bytes: bvh.memory_bytes(),
        upload: Duration::ZERO,
    };
    Ok(Prepared {
        mesh,
        gpu_data,
        bvh,
        info,
    })
}

/// Segments around the tube for a torus with `rings` segments around its axis.
pub fn torus_sides(rings: u32) -> u32 {
    (rings / 2).max(3)
}

pub fn display_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}
