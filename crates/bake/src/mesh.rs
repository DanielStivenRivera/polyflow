//! The low-poly mesh as the baker sees it: triangles with per-corner
//! position, normal, UV and MikkTSpace tangent.
//!
//! Engines rebuild the tangent basis from the mesh they import, so a normal
//! map only looks right if the baker used the very same triangles, normals
//! and tangents. Polyflow therefore fixes all three here, and exports the
//! same triangulated mesh with these normals for the engine:
//! - Triangles: quads are split along their shorter diagonal, other polygons
//!   are fanned from their first corner.
//! - Normals: smooth vertex normals, each face contributing its normal
//!   weighted by its corner angle (computed on polygons, so they do not depend
//!   on the triangulation).
//! - Tangents: MikkTSpace, through `bevy_mikktspace`, which reproduces the
//!   reference C implementation bit for bit.

use polyflow_core::glam::{Vec2, Vec3, Vec4};
use polyflow_core::halfedge::{FaceId, HalfEdgeMesh};

use crate::tangent::{self, TangentFrame};

/// One corner of a triangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    pub position: Vec3,
    /// Unit length.
    pub normal: Vec3,
    pub uv: Vec2,
    /// MikkTSpace tangent (unit length) and bitangent sign in `w`.
    pub tangent: Vec4,
}

/// A point on the low-poly surface, interpolated from a triangle's corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSample {
    pub position: Vec3,
    pub frame: TangentFrame,
}

/// Triangulated low-poly mesh with everything baking needs.
#[derive(Debug, Clone, Default)]
pub struct BakeMesh {
    /// Three corners per triangle.
    pub corners: Vec<Corner>,
    /// The retopology face each triangle comes from.
    pub source_faces: Vec<FaceId>,
    /// Faces left out because some corner has no UV.
    pub faces_without_uvs: usize,
}

impl BakeMesh {
    /// Builds the bake mesh from the retopology. Faces without UVs are left
    /// out (and counted): there is nowhere to bake them to.
    pub fn from_retopo(mesh: &HalfEdgeMesh) -> Self {
        let normals = vertex_normals(mesh);
        let mut out = BakeMesh::default();
        let mut polygon: Vec<Corner> = Vec::new();
        for f in mesh.faces() {
            polygon.clear();
            for h in mesh.face_halfedges(f) {
                let Some(uv) = mesh.uv(h) else { break };
                let v = mesh.origin(h);
                polygon.push(Corner {
                    position: mesh.position(v),
                    normal: normals[v.index()],
                    uv,
                    tangent: Vec4::ZERO,
                });
            }
            if polygon.len() != mesh.face_halfedges(f).count() {
                out.faces_without_uvs += 1;
                continue;
            }
            for [a, b, c] in triangulate(&polygon) {
                out.corners
                    .extend_from_slice(&[polygon[a], polygon[b], polygon[c]]);
                out.source_faces.push(f);
            }
        }
        tangent::generate(&mut out.corners);
        out
    }

    pub fn triangle_count(&self) -> usize {
        self.corners.len() / 3
    }

    pub fn triangle(&self, t: usize) -> &[Corner; 3] {
        self.corners[t * 3..t * 3 + 3]
            .try_into()
            .expect("three corners")
    }

    /// The surface at barycentric coordinates `bary` (weights of corners 1
    /// and 2; corner 0 gets the rest) of triangle `t`. Normal and tangent are
    /// interpolated without normalizing, as a MikkTSpace shader does.
    pub fn sample(&self, t: usize, bary: [f32; 2]) -> SurfaceSample {
        let [a, b, c] = self.triangle(t);
        let w = [1.0 - bary[0] - bary[1], bary[0], bary[1]];
        let mix3 = |f: fn(&Corner) -> Vec3| f(a) * w[0] + f(b) * w[1] + f(c) * w[2];
        let tangent = a.tangent * w[0] + b.tangent * w[1] + c.tangent * w[2];
        SurfaceSample {
            position: mix3(|c| c.position),
            frame: TangentFrame {
                normal: mix3(|c| c.normal),
                tangent: tangent.truncate(),
                sign: if tangent.w < 0.0 { -1.0 } else { 1.0 },
            },
        }
    }
}

/// Splits a polygon into triangles, as corner indices. See the module docs.
fn triangulate(polygon: &[Corner]) -> Vec<[usize; 3]> {
    let n = polygon.len();
    if n == 4 {
        let p = |i: usize| polygon[i].position;
        return if p(0).distance_squared(p(2)) <= p(1).distance_squared(p(3)) {
            vec![[0, 1, 2], [0, 2, 3]]
        } else {
            vec![[0, 1, 3], [1, 2, 3]]
        };
    }
    (1..n.saturating_sub(1)).map(|i| [0, i, i + 1]).collect()
}

/// Angle-weighted smooth normals, indexed by vertex slot.
fn vertex_normals(mesh: &HalfEdgeMesh) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; mesh.vertex_slots().count()];
    let mut points: Vec<Vec3> = Vec::new();
    for f in mesh.faces() {
        points.clear();
        points.extend(mesh.face_vertices(f).map(|v| mesh.position(v)));
        // Newell's method: robust for non-planar polygons.
        let n = points.len();
        let face_normal = (0..n)
            .map(|i| points[i].cross(points[(i + 1) % n]))
            .sum::<Vec3>()
            .normalize_or_zero();
        for (i, v) in mesh.face_vertices(f).enumerate() {
            let p = points[i];
            let to_prev = (points[(i + n - 1) % n] - p).normalize_or_zero();
            let to_next = (points[(i + 1) % n] - p).normalize_or_zero();
            let angle = to_prev.dot(to_next).clamp(-1.0, 1.0).acos();
            normals[v.index()] += face_normal * angle;
        }
    }
    for n in &mut normals {
        *n = n.normalize_or(Vec3::Z);
    }
    normals
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An `n` x `n` grid of quads on the XY plane, UVs equal to XY / n.
    pub(crate) fn uv_grid(n: usize, flip_u: bool) -> HalfEdgeMesh {
        let mut mesh = HalfEdgeMesh::new();
        let mut ids = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                ids.push(mesh.add_vertex(Vec3::new(i as f32, j as f32, 0.0)));
            }
        }
        let uv = |i: usize, j: usize| {
            let u = i as f32 / n as f32;
            Vec2::new(if flip_u { 1.0 - u } else { u }, j as f32 / n as f32)
        };
        for j in 0..n {
            for i in 0..n {
                let k = j * (n + 1) + i;
                let f = mesh
                    .add_face(&[ids[k], ids[k + 1], ids[k + n + 2], ids[k + n + 1]])
                    .unwrap();
                mesh.set_face_uvs(f, &[uv(i, j), uv(i + 1, j), uv(i + 1, j + 1), uv(i, j + 1)]);
            }
        }
        mesh
    }

    #[test]
    fn flat_grid_has_a_flat_frame() {
        for flip in [false, true] {
            let bake = BakeMesh::from_retopo(&uv_grid(3, flip));
            assert_eq!(bake.triangle_count(), 18);
            for corner in &bake.corners {
                assert!(corner.normal.distance(Vec3::Z) < 1e-6);
                // U runs along +X, or -X when mirrored; V along +Y either way.
                let expected = if flip { -Vec3::X } else { Vec3::X };
                assert!(
                    corner.tangent.truncate().distance(expected) < 1e-5,
                    "{flip}: {}",
                    corner.tangent
                );
                // bitangent = sign * cross(N, T) must be +Y.
                let bitangent = corner.tangent.w * corner.normal.cross(corner.tangent.truncate());
                assert!(bitangent.distance(Vec3::Y) < 1e-5);
            }
            let s = bake.sample(4, [0.25, 0.25]);
            assert_eq!(s.frame.sign, if flip { -1.0 } else { 1.0 });
        }
    }

    #[test]
    fn faces_without_uvs_are_skipped() {
        let mut mesh = uv_grid(2, false);
        let f = mesh.faces().next().unwrap();
        let h = mesh.face_halfedge(f);
        mesh.set_uv(h, None);
        let bake = BakeMesh::from_retopo(&mesh);
        assert_eq!(bake.faces_without_uvs, 1);
        assert_eq!(bake.triangle_count(), 6);
        assert!(bake.source_faces.iter().all(|&s| s != f));
    }

    #[test]
    fn quads_split_on_the_shorter_diagonal() {
        let corner = |x: f32, y: f32| Corner {
            position: Vec3::new(x, y, 0.0),
            normal: Vec3::Z,
            uv: Vec2::ZERO,
            tangent: Vec4::ZERO,
        };
        // A rhombus whose 1-3 diagonal is the short one.
        let quad = [
            corner(-2.0, 0.0),
            corner(0.0, -1.0),
            corner(2.0, 0.0),
            corner(0.0, 1.0),
        ];
        assert_eq!(triangulate(&quad), vec![[0, 1, 3], [1, 2, 3]]);
        assert_eq!(triangulate(&quad[..3]), vec![[0, 1, 2]]);
        let pentagon = [quad[0], quad[1], quad[2], quad[3], corner(-1.0, 1.0)];
        assert_eq!(triangulate(&pentagon).len(), 3);
    }

    #[test]
    fn normals_are_angle_weighted() {
        // A vertex shared by a quad on the XY plane and a triangle tilted up:
        // the quad's 90° corner outweighs the triangle's narrow one.
        let mut mesh = HalfEdgeMesh::new();
        let v = [
            Vec3::ZERO,
            Vec3::X,
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::Y,
            Vec3::new(3.0, -0.3, -0.3),
        ]
        .map(|p| mesh.add_vertex(p));
        mesh.add_face(&[v[0], v[1], v[2], v[3]]).unwrap();
        mesh.add_face(&[v[0], v[4], v[1]]).unwrap();
        let normals = vertex_normals(&mesh);
        let n = normals[v[0].index()];
        assert!((n.length() - 1.0).abs() < 1e-6);
        // An unweighted average of +Z and the triangle's (0, -1, 1) / √2
        // would give z ≈ 0.92.
        assert!(n.z > 0.99 && n.y < 0.0, "{n}");
    }
}
