//! Import and export of meshes and images, and the binary cache used to
//! reopen huge reference meshes quickly.
//!
//! Importers must treat every input file as untrusted: malformed data returns
//! an error, never panics.

#![forbid(unsafe_code)]

pub mod cache;
pub mod fbx;
pub mod gltf;
pub mod image;
pub mod lowpoly;
pub mod obj;
pub mod ply;
pub mod project;
pub mod retopo;

use std::path::Path;
use std::time::{Duration, Instant};

use polyflow_core::glam::Vec3;
use polyflow_core::mesh::TriMesh;

/// File extensions of the reference mesh formats [`load_mesh`] reads.
pub const MESH_EXTENSIONS: &[&str] = &["obj", "ply", "gltf", "glb", "fbx"];

/// Triangles read from a mesh file, before normals are computed.
#[derive(Debug, Default, PartialEq)]
pub struct Geometry {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    /// Faces with fewer than three vertices, which were skipped.
    pub skipped_faces: usize,
}

impl Geometry {
    /// Converts into a mesh, computing smooth vertex normals.
    pub fn into_mesh(self) -> TriMesh {
        let mut mesh = TriMesh {
            positions: self.positions,
            normals: vec![],
            indices: self.indices,
        };
        mesh.compute_normals();
        mesh
    }

    /// Appends a polygon (indices into `positions`) as a triangle fan;
    /// polygons with fewer than three corners count as skipped.
    pub(crate) fn push_polygon(&mut self, corners: &[u32]) {
        if corners.len() < 3 {
            self.skipped_faces += 1;
            return;
        }
        for i in 1..corners.len() - 1 {
            self.indices
                .extend_from_slice(&[corners[0], corners[i], corners[i + 1]]);
        }
    }
}

/// Errors from loading a mesh file.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("invalid OBJ file: {0}")]
    Obj(#[from] obj::ObjError),
    #[error("invalid PLY file: {0}")]
    Ply(#[from] ply::PlyError),
    #[error("invalid glTF file: {0}")]
    Gltf(#[from] gltf::GltfError),
    #[error("could not read the FBX file: {0}")]
    Fbx(String),
    #[error("unsupported file type `.{0}` (supported: .obj, .ply, .gltf, .glb, .fbx)")]
    UnsupportedFormat(String),
    #[error("the file contains no triangles")]
    Empty,
}

/// A loaded mesh with timings, for display and benchmarking.
#[derive(Debug)]
pub struct LoadedMesh {
    pub mesh: TriMesh,
    pub skipped_faces: usize,
    pub file_bytes: u64,
    pub read_time: Duration,
    pub parse_time: Duration,
    pub normals_time: Duration,
}

/// Writes a file through a temporary one next to it, renamed into place once
/// complete, so a failure never leaves a half-written file behind.
pub(crate) fn save_atomically(
    path: &Path,
    write: impl FnOnce(&std::fs::File) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let mut temp = path.as_os_str().to_owned();
    temp.push(".tmp");
    let temp = std::path::PathBuf::from(temp);
    let result = std::fs::File::create(&temp).and_then(|file| {
        write(&file)?;
        file.sync_all()
    });
    match result.and_then(|()| std::fs::rename(&temp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&temp);
            Err(e)
        }
    }
}

/// Loads a reference mesh, choosing the format from the file extension.
pub fn load_mesh(path: &Path) -> Result<LoadedMesh, LoadError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !MESH_EXTENSIONS.contains(&extension.as_str()) {
        return Err(LoadError::UnsupportedFormat(extension));
    }

    let start = Instant::now();
    let bytes = std::fs::read(path)?;
    let read_time = start.elapsed();

    let start = Instant::now();
    let geometry = match extension.as_str() {
        "obj" => obj::parse(&bytes)?,
        "ply" => ply::parse(&bytes)?,
        "gltf" | "glb" => gltf::parse(&bytes, path.parent())?,
        _ => fbx::parse(&bytes)?,
    };
    let parse_time = start.elapsed();
    let file_bytes = bytes.len() as u64;
    drop(bytes);
    if geometry.indices.is_empty() {
        return Err(LoadError::Empty);
    }
    let skipped_faces = geometry.skipped_faces;

    let start = Instant::now();
    let mesh = geometry.into_mesh();
    let normals_time = start.elapsed();

    Ok(LoadedMesh {
        mesh,
        skipped_faces,
        file_bytes,
        read_time,
        parse_time,
        normals_time,
    })
}
