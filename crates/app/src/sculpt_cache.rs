//! The sculpt cache: reference meshes as prepared for display and queries
//! (mesh, meshlets, BVH), so huge sculpts reopen without parsing their file,
//! clusterizing and building the BVH again.
//!
//! One cache file per sculpt, in the user's cache folder, used while the
//! sculpt's size and modification time are unchanged and by the same version
//! of Polyflow. The least recently used are deleted beyond [`MAX_BYTES`].

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use polyflow_core::mesh::TriMesh;
use polyflow_io::cache::{self, CacheError, SourceKey};
use polyflow_render::MeshData;
use polyflow_spatial::Bvh;

/// Total size the cache may take.
pub const MAX_BYTES: u64 = 8 << 30;

/// The folder holding the cache files.
pub fn dir() -> Option<PathBuf> {
    dirs::cache_dir().map(|d| d.join("Polyflow").join("sculpts"))
}

fn key(source: &Path) -> Option<SourceKey> {
    SourceKey::of(source, env!("CARGO_PKG_VERSION")).ok()
}

/// A sculpt read from the cache.
#[derive(Debug)]
pub struct Cached {
    pub mesh: TriMesh,
    pub gpu_data: MeshData,
    pub bvh: Bvh,
    /// Size of the cache file.
    pub bytes: u64,
    pub read: Duration,
}

/// The cached preparation of `source`, if there is a valid one. A damaged
/// cache file is deleted (and the sculpt prepared again from its file).
pub fn load(source: &Path) -> Option<Cached> {
    let key = key(source)?;
    let path = dir()?.join(key.file_name());
    let start = Instant::now();
    match read(&path, &key) {
        Ok(Some(mut cached)) => {
            cached.read = start.elapsed();
            log::info!(
                "Read {} from the sculpt cache in {:?} ({} MB)",
                source.display(),
                cached.read,
                cached.bytes >> 20
            );
            cache::touch(&path);
            Some(cached)
        }
        Ok(None) => None,
        Err(e) => {
            log::warn!("Ignoring the cache of {}: {e}", source.display());
            let _ = std::fs::remove_file(&path);
            None
        }
    }
}

fn read(path: &Path, key: &SourceKey) -> Result<Option<Cached>, CacheError> {
    let Some(file) = cache::open(path, key)? else {
        return Ok(None);
    };
    let invalid = CacheError::Invalid;
    // Sections are read in parallel: one large sequential read each keeps
    // several requests in flight, which fast SSDs need for full speed.
    let ((positions, normals), ((indices, meshlet_indices), (meshlets, (nodes, order)))) =
        rayon::join(
            || rayon::join(|| file.read("positions"), || file.read("normals")),
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || file.read::<u32>("indices"),
                            || file.read("meshlet_indices"),
                        )
                    },
                    || {
                        rayon::join(
                            || file.read("meshlets"),
                            || {
                                rayon::join(
                                    || file.read("bvh_nodes"),
                                    || file.read("bvh_triangles"),
                                )
                            },
                        )
                    },
                )
            },
        );
    let mesh = TriMesh {
        positions: positions?,
        normals: normals?,
        indices: indices?,
    };
    validate(&mesh).map_err(invalid)?;
    let gpu_data = MeshData::from_parts(&mesh, meshlet_indices?, meshlets?).map_err(invalid)?;
    let bvh = Bvh::from_parts(nodes?, order?, mesh.triangle_count()).map_err(invalid)?;
    Ok(Some(Cached {
        mesh,
        gpu_data,
        bvh,
        bytes: file.len,
        read: Duration::ZERO,
    }))
}

/// Checks what every user of a [`TriMesh`] relies on.
fn validate(mesh: &TriMesh) -> Result<(), String> {
    use rayon::prelude::*;
    if mesh.positions.len() > u32::MAX as usize || mesh.normals.len() != mesh.positions.len() {
        return Err("the vertices do not match".into());
    }
    if !mesh.indices.len().is_multiple_of(3) {
        return Err("a triangle is incomplete".into());
    }
    let count = mesh.positions.len();
    if mesh.indices.par_iter().any(|&i| i as usize >= count) {
        return Err("an index is out of range".into());
    }
    Ok(())
}

/// Writes the cache of `source` on a background thread, then trims the cache
/// to [`MAX_BYTES`]. Failures are only logged: the cache is an optimization.
pub fn save_in_background(
    source: PathBuf,
    mesh: Arc<TriMesh>,
    gpu_data: Arc<MeshData>,
    bvh: Arc<Bvh>,
) {
    let spawned = std::thread::Builder::new()
        .name("sculpt cache".into())
        .spawn(move || {
            let start = Instant::now();
            match save(&source, &mesh, &gpu_data, &bvh) {
                Ok(path) => log::info!(
                    "Cached {} in {:?} ({})",
                    source.display(),
                    start.elapsed(),
                    path.display()
                ),
                Err(e) => log::warn!("Could not cache {}: {e}", source.display()),
            }
        });
    if let Err(e) = spawned {
        log::warn!("Could not start caching: {e}");
    }
}

fn save(source: &Path, mesh: &TriMesh, gpu_data: &MeshData, bvh: &Bvh) -> std::io::Result<PathBuf> {
    let key = key(source).ok_or_else(|| std::io::Error::other("the sculpt's file is gone"))?;
    let dir = dir().ok_or_else(|| std::io::Error::other("no cache folder"))?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(key.file_name());
    write(&path, &key, mesh, gpu_data, bvh)?;
    cache::prune(&dir, MAX_BYTES, &path)?;
    Ok(path)
}

fn write(
    path: &Path,
    key: &SourceKey,
    mesh: &TriMesh,
    gpu_data: &MeshData,
    bvh: &Bvh,
) -> std::io::Result<()> {
    let (meshlet_indices, meshlets) = gpu_data.parts();
    let (nodes, order) = bvh.parts();
    cache::write(
        path,
        key,
        &[
            ("positions", bytemuck::cast_slice(&mesh.positions)),
            ("normals", bytemuck::cast_slice(&mesh.normals)),
            ("indices", bytemuck::cast_slice(&mesh.indices)),
            ("meshlet_indices", bytemuck::cast_slice(meshlet_indices)),
            ("meshlets", bytemuck::cast_slice(meshlets)),
            ("bvh_nodes", bytemuck::cast_slice(nodes)),
            ("bvh_triangles", bytemuck::cast_slice(order)),
        ],
    )
}

/// Size of the cache.
pub fn size() -> u64 {
    dir().map_or(0, |d| cache::size(&d))
}

/// Deletes the whole cache.
pub fn clear() -> std::io::Result<()> {
    match dir() {
        Some(dir) if dir.exists() => cache::clear(&dir),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use polyflow_core::primitives::torus;

    use super::*;

    #[test]
    fn round_trip_through_a_cache_file() {
        let mesh = torus(1.0, 0.3, 48, 24);
        let gpu_data = MeshData::build(&mesh);
        let bvh = Bvh::build(&mesh);
        let dir =
            std::env::temp_dir().join(format!("polyflow-sculpt-cache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("torus.pfcache");
        let key = SourceKey {
            path: "torus.obj".into(),
            len: 1,
            modified: 2,
            version: "test".into(),
        };
        write(&path, &key, &mesh, &gpu_data, &bvh).unwrap();
        let cached = read(&path, &key).unwrap().unwrap();
        assert_eq!(cached.mesh, mesh);
        assert_eq!(cached.gpu_data.parts(), gpu_data.parts());
        assert_eq!(cached.bvh.parts(), bvh.parts());

        // Damage inside a section is caught by validation.
        let mut bytes = std::fs::read(&path).unwrap();
        let len = bytes.len();
        for b in &mut bytes[len - 64..] {
            *b = 0xFF;
        }
        std::fs::write(&path, &bytes).unwrap();
        assert!(read(&path, &key).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
