//! Queries against the reference surface.

use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::mesh::TriMesh;
use polyflow_spatial::Bvh;

use super::view::ViewInfo;

/// The reference mesh and its BVH.
#[derive(Debug, Clone, Copy)]
pub struct Surface<'a> {
    pub mesh: &'a TriMesh,
    pub bvh: &'a Bvh,
}

/// A point on the reference surface with its smooth normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePoint {
    pub position: Vec3,
    pub normal: Vec3,
}

impl Surface<'_> {
    /// The surface under a window pixel.
    pub fn raycast(&self, view: &ViewInfo, cursor: Vec2) -> Option<SurfacePoint> {
        let (origin, dir) = view.ray(cursor);
        let hit = self.bvh.raycast(self.mesh, origin, dir, f32::INFINITY)?;
        Some(SurfacePoint {
            position: origin + dir * hit.t,
            normal: self.normal(hit.triangle, hit.u, hit.v),
        })
    }

    /// The closest surface point to `p`.
    pub fn closest(&self, p: Vec3) -> Option<SurfacePoint> {
        let hit = self.bvh.closest_point(self.mesh, p, f32::INFINITY)?;
        Some(SurfacePoint {
            position: hit.point,
            normal: self.normal(hit.triangle, hit.u, hit.v),
        })
    }

    /// `p` snapped onto the surface, or unchanged if there is no surface.
    pub fn snap(&self, p: Vec3) -> Vec3 {
        self.closest(p).map_or(p, |s| s.position)
    }

    /// Whether `p` can be seen from `eye`, i.e. the surface does not hide it.
    /// Points lying on the surface count as visible.
    pub fn is_visible(&self, eye: Vec3, p: Vec3) -> bool {
        let offset = p - eye;
        let distance = offset.length();
        if distance <= 1e-9 {
            return true;
        }
        let t_max = distance - (distance * 2e-3).max(1e-6);
        self.bvh
            .raycast(self.mesh, eye, offset / distance, t_max)
            .is_none()
    }

    fn normal(&self, triangle: u32, u: f32, v: f32) -> Vec3 {
        let i = triangle as usize * 3;
        let [a, b, c] = [0, 1, 2].map(|k| self.mesh.indices[i + k] as usize);
        let n = if self.mesh.normals.len() == self.mesh.positions.len() {
            self.mesh.normals[a] * (1.0 - u - v)
                + self.mesh.normals[b] * u
                + self.mesh.normals[c] * v
        } else {
            let p = &self.mesh.positions;
            (p[b] - p[a]).cross(p[c] - p[a])
        };
        n.normalize_or(Vec3::Y)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use polyflow_core::primitives::torus;

    use super::*;

    /// A flat square in the XZ plane, facing +Y, of half-size `size`.
    pub fn plane(size: f32, cells: u32) -> TriMesh {
        let mut mesh = TriMesh::default();
        let n = cells + 1;
        for j in 0..n {
            for i in 0..n {
                let (x, z) = (
                    i as f32 / cells as f32 * 2.0 - 1.0,
                    j as f32 / cells as f32 * 2.0 - 1.0,
                );
                mesh.positions.push(Vec3::new(x * size, 0.0, z * size));
            }
        }
        for j in 0..cells {
            for i in 0..cells {
                let k = j * n + i;
                mesh.indices
                    .extend_from_slice(&[k, k + n, k + 1, k + 1, k + n, k + n + 1]);
            }
        }
        mesh.compute_normals();
        mesh
    }

    #[test]
    fn snapping_and_visibility() {
        let mesh = torus(1.0, 0.3, 64, 32);
        let bvh = Bvh::build(&mesh);
        let surface = Surface {
            mesh: &mesh,
            bvh: &bvh,
        };
        let s = surface.closest(Vec3::new(2.0, 0.0, 0.0)).unwrap();
        assert!((s.position - Vec3::new(1.3, 0.0, 0.0)).length() < 1e-2);
        assert!(s.normal.dot(Vec3::X) > 0.99);
        let eye = Vec3::new(5.0, 0.0, 0.0);
        assert!(
            surface.is_visible(eye, Vec3::new(1.3, 0.0, 0.0)),
            "front of the tube"
        );
        assert!(
            !surface.is_visible(eye, Vec3::new(0.7, 0.0, 0.0)),
            "back of the tube"
        );
        assert!(plane(1.0, 2).normals.iter().all(|n| n.dot(Vec3::Y) > 0.99));
    }
}
