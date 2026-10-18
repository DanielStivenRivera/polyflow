//! Procedural test meshes, used until real importers land and for benchmarks.

use glam::Vec3;

use crate::mesh::TriMesh;

/// A torus around the Y axis.
///
/// `rings` is the number of segments around the main circle and `sides` the
/// number around the tube; the result has `2 * rings * sides` triangles.
/// Both are clamped to at least 3.
pub fn torus(major_radius: f32, minor_radius: f32, rings: u32, sides: u32) -> TriMesh {
    let (rings, sides) = (rings.max(3), sides.max(3));
    let vertex_count = (rings * sides) as usize;
    let mut mesh = TriMesh {
        positions: Vec::with_capacity(vertex_count),
        normals: Vec::with_capacity(vertex_count),
        indices: Vec::with_capacity(vertex_count * 6),
    };

    for i in 0..rings {
        let (su, cu) = (i as f32 / rings as f32 * std::f32::consts::TAU).sin_cos();
        let ring_dir = Vec3::new(cu, 0.0, su);
        for j in 0..sides {
            let (sv, cv) = (j as f32 / sides as f32 * std::f32::consts::TAU).sin_cos();
            let normal = ring_dir * cv + Vec3::Y * sv;
            mesh.positions
                .push(ring_dir * major_radius + normal * minor_radius);
            mesh.normals.push(normal);
        }
    }

    let index = |i: u32, j: u32| (i % rings) * sides + (j % sides);
    for i in 0..rings {
        for j in 0..sides {
            let (a, b) = (index(i, j), index(i + 1, j));
            let (c, d) = (index(i + 1, j + 1), index(i, j + 1));
            mesh.indices.extend_from_slice(&[a, d, b, b, d, c]);
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torus_counts_and_indices() {
        let mesh = torus(1.0, 0.25, 32, 16);
        assert_eq!(mesh.positions.len(), 32 * 16);
        assert_eq!(mesh.normals.len(), mesh.positions.len());
        assert_eq!(mesh.triangle_count(), 2 * 32 * 16);
        assert!(
            mesh.indices
                .iter()
                .all(|&i| (i as usize) < mesh.positions.len())
        );
    }

    #[test]
    fn torus_winding_faces_outward() {
        let mesh = torus(1.0, 0.25, 24, 12);
        for tri in mesh.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(|i| mesh.positions[i as usize]);
            let face_normal = (b - a).cross(c - a);
            let vertex_normal = mesh.normals[tri[0] as usize];
            assert!(face_normal.dot(vertex_normal) > 0.0);
        }
    }

    #[test]
    fn torus_bounds() {
        let b = torus(2.0, 0.5, 64, 32).bounds();
        assert!((b.max.x - 2.5).abs() < 1e-4 && (b.max.y - 0.5).abs() < 1e-4);
    }
}
