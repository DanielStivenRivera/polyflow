//! Texture baking: projects detail from the reference mesh onto the
//! retopology mesh (tangent-space normals, AO, height, cavity, curvature).
//!
//! Contains a CPU reference implementation used to validate the GPU path.
//!
//! - [`mesh`]: the low-poly as baked (triangles, normals, MikkTSpace tangents).
//! - [`raster`]: which triangle covers each texel of a map, and where.
//! - [`tangent`]: tangent frames and the tangent-space encoding of normals.

#![forbid(unsafe_code)]

pub mod mesh;
pub mod raster;
pub mod tangent;

pub use mesh::{BakeMesh, Corner, SurfaceSample};
pub use raster::{UvRaster, rasterize};
pub use tangent::TangentFrame;
