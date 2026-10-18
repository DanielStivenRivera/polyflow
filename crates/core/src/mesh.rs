//! Indexed triangle meshes, used for the high-poly reference.

use glam::Vec3;
use rayon::prelude::*;

use crate::bounds::Aabb;

/// Work unit for parallel loops over vertices or triangles: large enough to
/// amortize scheduling, small enough to balance across cores.
const PAR_CHUNK: usize = 1 << 16;

/// An indexed triangle mesh with per-vertex normals.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TriMesh {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// Three indices per triangle, counter-clockwise when seen from outside.
    pub indices: Vec<u32>,
}

impl TriMesh {
    pub fn vertex_count(&self) -> usize {
        self.positions.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    pub fn bounds(&self) -> Aabb {
        self.positions
            .par_chunks(PAR_CHUNK)
            .map(Aabb::from_points)
            .reduce(|| Aabb::EMPTY, Aabb::union)
    }

    /// Replaces `normals` with smooth, area-weighted vertex normals.
    pub fn compute_normals(&mut self) {
        self.normals = vertex_normals(&self.positions, &self.indices);
    }
}

/// Area-weighted vertex normals. Vertices not used by any non-degenerate
/// triangle get `+Y`. Every index must be in bounds for `positions`.
pub fn vertex_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let triangles = indices.as_chunks::<3>().0;
    let mut normals = vec![Vec3::ZERO; positions.len()];
    // Face normals are computed in parallel into a bounded scratch buffer,
    // then scattered serially: the scatter has no safe parallel form without
    // per-thread copies of the whole normal array.
    const BATCH: usize = 1 << 20;
    let mut face_normals = Vec::with_capacity(BATCH.min(triangles.len()));
    for batch in triangles.chunks(BATCH) {
        batch
            .par_iter()
            .map(|&[a, b, c]| {
                let [a, b, c] = [a, b, c].map(|i| positions[i as usize]);
                // The cross product's length is twice the area: area weighting for free.
                (b - a).cross(c - a)
            })
            .collect_into_vec(&mut face_normals);
        for (tri, &n) in batch.iter().zip(&face_normals) {
            for &i in tri {
                normals[i as usize] += n;
            }
        }
    }
    normals.par_chunks_mut(PAR_CHUNK).for_each(|chunk| {
        for n in chunk {
            let len = n.length();
            *n = if len > 0.0 && len.is_finite() {
                *n / len
            } else {
                Vec3::Y
            };
        }
    });
    normals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::torus;

    #[test]
    fn normals_of_a_flat_quad_point_up() {
        let positions = [
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(1.0, 0.0, -1.0),
            Vec3::NEG_Z,
            Vec3::splat(9.0),
        ];
        let indices = [0, 1, 2, 0, 2, 3];
        let normals = vertex_normals(&positions, &indices);
        for n in &normals[..4] {
            assert!((*n - Vec3::Y).length() < 1e-6, "{n}");
        }
        assert_eq!(normals[4], Vec3::Y, "unused vertex gets the fallback");
    }

    #[test]
    fn degenerate_triangles_do_not_produce_nan() {
        let positions = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let normals = vertex_normals(&positions, &[0, 1, 2]);
        assert!(normals.iter().all(|n| n.is_finite()));
    }

    #[test]
    fn computed_normals_match_analytic_torus_normals() {
        let mut mesh = torus(1.0, 0.3, 64, 32);
        let analytic = mesh.normals.clone();
        mesh.compute_normals();
        for (a, b) in analytic.iter().zip(&mesh.normals) {
            assert!(a.dot(*b) > 0.99, "{a} vs {b}");
        }
    }

    #[test]
    fn bounds_match_serial() {
        let mesh = torus(2.0, 0.5, 300, 200);
        assert_eq!(mesh.bounds(), Aabb::from_points(&mesh.positions));
    }
}
