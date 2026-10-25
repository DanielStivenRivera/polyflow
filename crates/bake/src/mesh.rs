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
//!   on the triangulation), split across hard edges: edges marked hard, and
//!   UV seams if [`NormalOptions::hard_uv_seams`]. Each side of a hard edge
//!   averages only the faces on its side.
//! - Tangents: MikkTSpace, through `bevy_mikktspace`, which reproduces the
//!   reference C implementation bit for bit.
//!
//! Rays are cast along the smooth normals, split or not (as with "averaged
//! normals" in other bakers): split normals would aim the texels on both
//! sides of a hard edge away from each other and leave a gap of the
//! reference unbaked between them.

use polyflow_core::glam::{Vec2, Vec3, Vec4};
use polyflow_core::halfedge::{FaceId, HalfEdgeId, HalfEdgeMesh};

use crate::cage::Cage;
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
    /// Where rays start, if the mesh has a cage (see [`BakeMesh::has_cage`]).
    pub cage: Vec3,
    /// The smooth vertex normal (unit length), never split: rays without a
    /// cage are cast along it.
    pub projection: Vec3,
}

/// A point on the low-poly surface, interpolated from a triangle's corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSample {
    pub position: Vec3,
    pub frame: TangentFrame,
    /// The matching point on the cage (the position itself without one).
    pub cage: Vec3,
    /// Interpolated [`Corner::projection`], not normalized.
    pub projection: Vec3,
}

/// How the low-poly's normals are split. Edges marked hard always split them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NormalOptions {
    /// Also split normals where UV islands meet.
    pub hard_uv_seams: bool,
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
    /// Whether [`Corner::cage`] holds cage positions.
    pub has_cage: bool,
}

impl BakeMesh {
    /// Builds the bake mesh from the retopology. Faces without UVs are left
    /// out (and counted): there is nowhere to bake them to.
    pub fn from_retopo(mesh: &HalfEdgeMesh) -> Self {
        Self::with_cage(mesh, None)
    }

    /// Like [`Self::from_retopo`], with rays starting from `cage`. Vertices
    /// the cage does not cover start from themselves.
    pub fn with_cage(mesh: &HalfEdgeMesh, cage: Option<&Cage>) -> Self {
        Self::new(mesh, cage, NormalOptions::default())
    }

    /// Like [`Self::with_cage`], with normals split as `options` says.
    pub fn new(mesh: &HalfEdgeMesh, cage: Option<&Cage>, options: NormalOptions) -> Self {
        let smooth = vertex_normals(mesh);
        let normals = corner_normals(mesh, options);
        let mut out = BakeMesh {
            has_cage: cage.is_some(),
            ..BakeMesh::default()
        };
        let mut polygon: Vec<Corner> = Vec::new();
        for f in mesh.faces() {
            polygon.clear();
            for h in mesh.face_halfedges(f) {
                let Some(uv) = mesh.uv(h) else { break };
                let v = mesh.origin(h);
                let position = mesh.position(v);
                polygon.push(Corner {
                    position,
                    normal: normals[h.index()],
                    uv,
                    tangent: Vec4::ZERO,
                    cage: cage.and_then(|c| c.position(v)).unwrap_or(position),
                    projection: smooth[v.index()],
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
            cage: mix3(|c| c.cage),
            projection: mix3(|c| c.projection),
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
pub(crate) fn vertex_normals(mesh: &HalfEdgeMesh) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; mesh.vertex_slots().count()];
    for_each_weighted_corner(mesh, |h, normal| {
        normals[mesh.origin(h).index()] += normal;
    });
    for n in &mut normals {
        *n = n.normalize_or(Vec3::Z);
    }
    normals
}

/// Angle-weighted normals of the face corners, indexed by half-edge slot
/// (the corner at a half-edge's origin): the corners around a vertex are
/// averaged together, except across hard edges, which split them into
/// groups averaged apart. Without hard edges these are [`vertex_normals`],
/// bit for bit (the same sums in the same order).
pub(crate) fn corner_normals(mesh: &HalfEdgeMesh, options: NormalOptions) -> Vec<Vec3> {
    let slots = mesh.halfedges().last().map_or(0, |h| h.index() + 1);
    // Union-find over corners; corners on either side of a smooth edge, at
    // the same end of it, share a group.
    let mut parent: Vec<u32> = (0..slots as u32).collect();
    fn root(parent: &mut [u32], h: usize) -> usize {
        let mut r = h;
        while parent[r] as usize != r {
            r = parent[r] as usize;
        }
        let mut h = h;
        while parent[h] as usize != r {
            let next = parent[h] as usize;
            parent[h] = r as u32;
            h = next;
        }
        r
    }
    let join = |parent: &mut Vec<u32>, a: HalfEdgeId, b: HalfEdgeId| {
        let (ra, rb) = (root(parent, a.index()), root(parent, b.index()));
        // The smaller slot is the root, so groups do not depend on the order
        // edges are visited in.
        let (lo, hi) = (ra.min(rb), ra.max(rb));
        parent[hi] = lo as u32;
    };
    for e in mesh.edges() {
        let (h, t) = (e.halfedge(0), e.halfedge(1));
        if mesh.face_of(h).is_none() || mesh.face_of(t).is_none() || mesh.is_hard(e) {
            continue;
        }
        // `h` and `next(t)` are the corners at `h`'s origin in its two faces,
        // `next(h)` and `t` those at its target.
        let (start, end) = ((h, mesh.next(t)), (mesh.next(h), t));
        if options.hard_uv_seams
            && (mesh.uv(start.0) != mesh.uv(start.1) || mesh.uv(end.0) != mesh.uv(end.1))
        {
            continue;
        }
        join(&mut parent, start.0, start.1);
        join(&mut parent, end.0, end.1);
    }
    let mut sums = vec![Vec3::ZERO; slots];
    for_each_weighted_corner(mesh, |h, normal| {
        sums[root(&mut parent, h.index())] += normal;
    });
    (0..slots)
        .map(|h| sums[root(&mut parent, h)].normalize_or(Vec3::Z))
        .collect()
}

/// Calls `add` with each face corner (its half-edge) and the face's normal
/// weighted by the corner's angle, face by face.
fn for_each_weighted_corner(mesh: &HalfEdgeMesh, mut add: impl FnMut(HalfEdgeId, Vec3)) {
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
        for (i, h) in mesh.face_halfedges(f).enumerate() {
            let p = points[i];
            let to_prev = (points[(i + n - 1) % n] - p).normalize_or_zero();
            let to_next = (points[(i + 1) % n] - p).normalize_or_zero();
            let angle = to_prev.dot(to_next).clamp(-1.0, 1.0).acos();
            add(h, face_normal * angle);
        }
    }
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
            cage: Vec3::ZERO,
            projection: Vec3::Z,
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

    /// A cube's corner: three faces meeting at right angles, UV islands of
    /// their own.
    fn cube() -> HalfEdgeMesh {
        let mut mesh = HalfEdgeMesh::new();
        let v: Vec<_> = (0..8)
            .map(|i| {
                let bit = |b: u32| if i & (1 << b) != 0 { 1.0 } else { -1.0 };
                mesh.add_vertex(Vec3::new(bit(0), bit(1), bit(2)))
            })
            .collect();
        let quads = [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [1, 3, 7, 5],
        ];
        for (k, q) in quads.iter().enumerate() {
            let f = mesh.add_face(&q.map(|i| v[i])).unwrap();
            let o = Vec2::new(k as f32 * 0.2, 0.0);
            let uvs = [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y].map(|uv| o + uv * 0.1);
            mesh.set_face_uvs(f, &uvs);
        }
        mesh
    }

    #[test]
    fn hard_edges_split_normals() {
        let mut mesh = cube();
        let smooth = BakeMesh::from_retopo(&mesh);
        for c in &smooth.corners {
            // Every corner averages three faces: along the cube's diagonal.
            assert!(c.normal.distance(c.position.normalize()) < 1e-6);
            assert_eq!(c.normal, c.projection);
        }

        // UV seams everywhere: flat faces, rays still along the diagonals.
        let seams = NormalOptions {
            hard_uv_seams: true,
        };
        let flat = BakeMesh::new(&mesh, None, seams);
        for (t, &f) in flat.source_faces.iter().enumerate() {
            let face_normal = flat.triangle(t)[0].normal;
            for c in flat.triangle(t) {
                assert_eq!(c.normal, face_normal, "{f:?}");
                assert!(c.position.dot(face_normal) > 0.99);
                assert!(c.projection.distance(c.position.normalize()) < 1e-6);
            }
        }

        // Hard edges around the -z face: it turns flat, and the faces
        // around it average only among themselves. A single hard edge would
        // not split anything: the faces at its ends still connect around
        // the other way.
        let at = |p: Vec3| mesh.vertices().find(|&v| mesh.position(v) == p).unwrap();
        let ring = [(-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0), (1.0, -1.0)]
            .map(|(x, y)| at(Vec3::new(x, y, -1.0)));
        for i in 0..4 {
            let e = mesh
                .find_halfedge(ring[i], ring[(i + 1) % 4])
                .unwrap()
                .edge();
            mesh.set_hard(e, true);
        }
        let split = BakeMesh::from_retopo(&mesh);
        let corner = Vec3::splat(-1.0);
        let mut normals: Vec<Vec3> = split
            .corners
            .iter()
            .filter(|c| c.position == corner)
            .map(|c| c.normal)
            .collect();
        normals.sort_by(|p, q| p.to_array().partial_cmp(&q.to_array()).unwrap());
        normals.dedup_by(|p, q| p.distance(*q) < 1e-6);
        let s = 0.5f32.sqrt();
        assert_eq!(normals.len(), 2, "{normals:?}");
        assert!(
            normals[0].distance(Vec3::new(-s, -s, 0.0)) < 1e-6,
            "{normals:?}"
        );
        assert!(normals[1].distance(-Vec3::Z) < 1e-6, "{normals:?}");
        // Corners away from the edge are unchanged.
        let far = Vec3::ONE;
        assert!(
            split
                .corners
                .iter()
                .filter(|c| c.position == far)
                .all(|c| c.normal.distance(far.normalize()) < 1e-6)
        );
    }
}
