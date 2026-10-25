//! Spatial acceleration structures: BVH construction, raycasts and
//! closest-point queries against the reference mesh.
//!
//! Used on the CPU for latency-critical snapping, and as the source data for
//! GPU-side acceleration structures.

#![forbid(unsafe_code)]

mod bvh;
pub mod triangle;

pub use bvh::{Bvh, BvhNode, ClosestPoint, RayHit, STACK_SIZE};
