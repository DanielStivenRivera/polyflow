//! Texture baking: projects detail from the reference mesh onto the
//! retopology mesh (tangent-space normals, AO, height, cavity, curvature).
//!
//! Contains a CPU reference implementation used to validate the GPU path.
//!
//! - [`cage`]: where bake rays start: pushed out automatically or imported.
//! - [`cpu`]: the CPU reference baker (normals, ambient occlusion).
//! - [`mesh`]: the low-poly as baked (triangles, normals, MikkTSpace tangents).
//! - [`padding`]: extending UV islands into the background.
//! - [`raster`]: which triangle covers each texel of a map, and where.
//! - [`tangent`]: tangent frames and the tangent-space encoding of normals.

#![forbid(unsafe_code)]

pub mod cage;
pub mod cpu;
pub mod mesh;
pub mod padding;
pub mod raster;
pub mod tangent;

pub use cage::{Cage, CageMismatch};
pub use cpu::{
    BakeControl, BakeError, BakeSettings, BakeSums, Baked, Coverage, MAX_SAMPLES, Map, MapSet,
    NormalY, Reference, bake, finish,
};
pub use mesh::{BakeMesh, Corner, SurfaceSample};
pub use raster::{UvRaster, rasterize, rasterize_rows};
pub use tangent::TangentFrame;
