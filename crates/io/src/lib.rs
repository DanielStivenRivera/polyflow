//! Import and export of meshes and images, plus (later) the fast binary cache
//! used to reopen huge reference meshes quickly.
//!
//! Importers must treat every input file as untrusted: malformed data returns
//! an error, never panics.

#![forbid(unsafe_code)]

pub mod image;
pub mod lowpoly;
pub mod obj;
pub mod project;
pub mod retopo;

use std::path::Path;
use std::time::{Duration, Instant};

use polyflow_core::mesh::TriMesh;

/// Errors from loading a mesh file.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("invalid OBJ file: {0}")]
    Obj(#[from] obj::ObjError),
    #[error("unsupported file type `.{0}` (supported: .obj)")]
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

/// Loads a reference mesh, choosing the format from the file extension.
pub fn load_mesh(path: &Path) -> Result<LoadedMesh, LoadError> {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if extension != "obj" {
        return Err(LoadError::UnsupportedFormat(extension));
    }

    let start = Instant::now();
    let bytes = std::fs::read(path)?;
    let read_time = start.elapsed();

    let start = Instant::now();
    let geometry = obj::parse(&bytes)?;
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
