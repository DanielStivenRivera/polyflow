use std::collections::HashSet;

use glam::{Vec2, Vec3};
use proptest::prelude::*;

use super::*;
use crate::history::MeshDocument;

/// Small deterministic PRNG for shuffles.
struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }
}

/// `(n + 1)^2` vertices on the XY plane and the counter-clockwise quad of each cell.
fn grid(n: usize) -> (HalfEdgeMesh, Vec<[VertexId; 4]>) {
    let mut mesh = HalfEdgeMesh::new();
    let mut ids = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            ids.push(mesh.add_vertex(Vec3::new(i as f32, j as f32, 0.0)));
        }
    }
    let at = |i: usize, j: usize| ids[j * (n + 1) + i];
    let mut cells = Vec::new();
    for j in 0..n {
        for i in 0..n {
            cells.push([at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
        }
    }
    (mesh, cells)
}

fn add_all(mesh: &mut HalfEdgeMesh, faces: &[[VertexId; 4]]) {
    for face in faces {
        mesh.add_face(face)
            .unwrap_or_else(|e| panic!("adding {face:?}: {e}"));
        mesh.validate()
            .unwrap_or_else(|e| panic!("after adding {face:?}: {e}"));
    }
}

fn boundary_vertices(mesh: &HalfEdgeMesh) -> usize {
    mesh.vertices()
        .filter(|&v| mesh.is_boundary_vertex(v))
        .count()
}

#[test]
fn grid_in_any_order() {
    let n = 6;
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let (_, cells) = grid(n);
    let mut orders = vec![cells.clone()];
    orders.push(cells.iter().rev().copied().collect());
    // Checkerboard: every second cell first, so the rest close holes whose
    // corners already have several fans, which exercises patch relinking.
    let (even, odd): (Vec<_>, Vec<_>) = cells
        .iter()
        .enumerate()
        .partition(|(k, _)| (k % n + k / n) % 2 == 0);
    orders.push(even.iter().chain(&odd).map(|(_, c)| **c).collect());
    for _ in 0..20 {
        let mut shuffled = cells.clone();
        rng.shuffle(&mut shuffled);
        orders.push(shuffled);
    }
    for order in orders {
        let (mut mesh, _) = grid(n);
        add_all(&mut mesh, &order);
        assert_eq!(mesh.face_count(), n * n);
        assert_eq!(mesh.edge_count(), 2 * n * (n + 1));
        assert_eq!(boundary_vertices(&mesh), 4 * n);
        assert_eq!(mesh.vertex_faces(VertexId(((n + 1) + 1) as u32)).count(), 4);
    }
}

#[test]
fn fan_closes_in_any_order() {
    let mut rng = Rng(99);
    for _ in 0..50 {
        let mut mesh = HalfEdgeMesh::new();
        let center = mesh.add_vertex(Vec3::ZERO);
        let ring: Vec<VertexId> = (0..7)
            .map(|k| {
                let a = k as f32 / 7.0 * std::f32::consts::TAU;
                mesh.add_vertex(Vec3::new(a.cos(), a.sin(), 0.0))
            })
            .collect();
        let mut faces: Vec<usize> = (0..7).collect();
        rng.shuffle(&mut faces);
        for k in faces {
            mesh.add_face(&[center, ring[k], ring[(k + 1) % 7]])
                .unwrap();
            mesh.validate().unwrap();
        }
        assert!(!mesh.is_boundary_vertex(center));
        assert_eq!(mesh.vertex_neighbors(center).count(), 7);
    }
}

#[test]
fn closed_cube() {
    let mut rng = Rng(7);
    let corners = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ];
    // Outward-facing, counter-clockwise.
    let quads = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    for _ in 0..30 {
        let mut mesh = HalfEdgeMesh::new();
        let v: Vec<VertexId> = corners
            .iter()
            .map(|c| mesh.add_vertex(Vec3::from_array(c.map(|x| x as f32))))
            .collect();
        let mut order = quads;
        rng.shuffle(&mut order);
        add_all(&mut mesh, &order.map(|q| q.map(|i| v[i])));
        assert_eq!(
            mesh.halfedges()
                .filter(|&h| mesh.is_boundary_halfedge(h))
                .count(),
            0
        );
        let euler =
            mesh.vertex_count() as i64 - mesh.edge_count() as i64 + mesh.face_count() as i64;
        assert_eq!(euler, 2);
    }
}

#[test]
fn faces_touching_at_a_vertex() {
    let mut mesh = HalfEdgeMesh::new();
    let c = mesh.add_vertex(Vec3::ZERO);
    let p = [(1.0, 0.0), (1.0, 1.0), (-1.0, 0.0), (-1.0, -1.0)]
        .map(|(x, y)| mesh.add_vertex(Vec3::new(x, y, 0.0)));
    mesh.add_face(&[c, p[0], p[1]]).unwrap();
    mesh.add_face(&[c, p[2], p[3]]).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.vertex_neighbors(c).count(), 4);
    // Bridging the two fans makes the vertex a single fan again.
    mesh.add_face(&[c, p[1], p[2]]).unwrap();
    mesh.validate().unwrap();
}

#[test]
fn invalid_faces_are_refused_without_changes() {
    let (mut mesh, cells) = grid(2);
    add_all(&mut mesh, &cells);
    let before = mesh.clone();
    let [a, b, c, d] = cells[0];
    let interior = VertexId(4); // Center of the 2x2 grid.
    let far = mesh.add_vertex(Vec3::new(5.0, 5.0, 0.0));
    let snapshot = mesh.clone();
    let cases: [(&[VertexId], TopologyError); 6] = [
        (&[a, b], TopologyError::TooFewVertices),
        (&[a, b, a], TopologyError::RepeatedVertex),
        (
            &[a, b, VertexId(999)],
            TopologyError::MissingVertex(VertexId(999)),
        ),
        (&[a, b, far], TopologyError::EdgeTaken(a, b)),
        (&[interior, far, d], TopologyError::InteriorVertex(interior)),
        (&[d, c, b, a], TopologyError::EdgeTaken(d, c)),
    ];
    for (face, expected) in cases {
        assert_eq!(mesh.add_face(face), Err(expected));
        assert_eq!(mesh, snapshot);
    }
    let _ = before;
}

#[test]
fn remove_and_restore_faces() {
    let (mut mesh, cells) = grid(3);
    add_all(&mut mesh, &cells);
    let full = mesh.clone();
    let center = mesh.faces().nth(4).unwrap();
    let center_vertices: Vec<VertexId> = mesh.face_vertices(center).collect();
    mesh.remove_face(center, false).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 8);
    assert_eq!(mesh.edge_count(), full.edge_count(), "edges of a hole stay");
    assert_eq!(boundary_vertices(&mesh), 12 + 4);
    mesh.add_face(&center_vertices).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 9);

    // Removing every face (keeping vertices) leaves no edges behind.
    let faces: Vec<FaceId> = mesh.faces().collect();
    for f in faces {
        mesh.remove_face(f, false).unwrap();
        mesh.validate().unwrap();
    }
    assert_eq!(
        (mesh.face_count(), mesh.edge_count(), mesh.vertex_count()),
        (0, 0, 16)
    );
}

#[test]
fn remove_vertices() {
    let (mut mesh, cells) = grid(3);
    add_all(&mut mesh, &cells);
    mesh.remove_vertex(VertexId(5)).unwrap(); // Interior: removes 4 faces.
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 5);
    mesh.remove_vertex(VertexId(3)).unwrap(); // Far corner: removes 1 face.
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 4);
    assert_eq!(
        mesh.remove_vertex(VertexId(3)),
        Err(TopologyError::MissingVertex(VertexId(3)))
    );
}

#[test]
fn triangulation() {
    let (mut mesh, cells) = grid(3);
    add_all(&mut mesh, &cells);
    let tris = mesh.triangulate();
    assert_eq!(tris.len(), 9 * 2 * 3);
    let positions: Vec<Option<Vec3>> = mesh.vertex_slots().collect();
    for t in tris.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| positions[i as usize].unwrap());
        assert!((b - a).cross(c - a).z > 0.0, "counter-clockwise from +Z");
    }
}

// ---- Property tests --------------------------------------------------------

const N: usize = 4;

#[derive(Debug, Clone)]
enum Op {
    Add {
        cell: usize,
        shape: u8,
        flip: bool,
    },
    RemoveFace(usize),
    RemoveVertex(usize),
    Move(usize, [i8; 3]),
    /// A topology operation on the k-th edge, vertex or face.
    Topology(u8, usize, usize),
    Undo,
    Redo,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (0..N * N, 0u8..3, prop::bool::weighted(0.1)).prop_map(|(cell, shape, flip)| Op::Add { cell, shape, flip }),
        1 => any::<usize>().prop_map(Op::RemoveFace),
        1 => any::<usize>().prop_map(Op::RemoveVertex),
        1 => (any::<usize>(), any::<[i8; 3]>()).prop_map(|(v, d)| Op::Move(v, d)),
        3 => (0u8..11, any::<usize>(), any::<usize>()).prop_map(|(kind, a, b)| Op::Topology(kind, a, b)),
        2 => Just(Op::Undo),
        1 => Just(Op::Redo),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, ..ProptestConfig::default() })]

    #[test]
    fn random_edits_keep_invariants_and_undo_exactly(ops in prop::collection::vec(op(), 1..80)) {
        let (mesh, cells) = grid(N);
        let initial = mesh.clone();
        let mut doc = MeshDocument::new(mesh);
        for op in ops {
            let before = doc.mesh().clone();
            let steps = doc.undo_steps();
            let result = match op {
                Op::Add { cell, shape, flip } => {
                    let [a, b, c, d] = cells[cell];
                    let mut face = match shape {
                        0 => vec![a, b, c, d],
                        1 => vec![a, b, c],
                        _ => vec![a, c, d],
                    };
                    if flip {
                        face.reverse();
                    }
                    doc.edit("Add face", |m| m.add_face(&face).map(|_| ()))
                }
                Op::RemoveFace(k) => {
                    let face = doc.mesh().faces().nth(k % doc.mesh().face_count().max(1));
                    match face {
                        Some(f) => doc.edit("Remove face", |m| m.remove_face(f, true)),
                        None => Ok(()),
                    }
                }
                Op::RemoveVertex(k) => {
                    let vertex = doc.mesh().vertices().nth(k % doc.mesh().vertex_count().max(1));
                    match vertex {
                        Some(v) => doc.edit("Remove vertex", |m| m.remove_vertex(v)),
                        None => Ok(()),
                    }
                }
                Op::Move(k, d) => {
                    let vertex = doc.mesh().vertices().nth(k % doc.mesh().vertex_count().max(1));
                    if let Some(v) = vertex {
                        let p = doc.mesh().position(v) + Vec3::new(d[0] as f32, d[1] as f32, d[2] as f32) * 0.01;
                        doc.edit("Move", |m| { m.set_position(v, p); Ok::<_, TopologyError>(()) }).unwrap();
                    }
                    Ok(())
                }
                Op::Topology(kind, a, b) => {
                    let mesh = doc.mesh();
                    let edge = mesh.halfedges().nth(a % mesh.halfedges().count().max(1));
                    let vertex = mesh.vertices().nth(a % mesh.vertex_count().max(1));
                    let other = mesh.vertices().nth(b % mesh.vertex_count().max(1));
                    let face = mesh.faces().nth(a % mesh.face_count().max(1));
                    let t = (b % 7) as f32 / 6.0;
                    let second = mesh.faces().nth(b % mesh.face_count().max(1));
                    let border = edge.filter(|&h| mesh.is_boundary_halfedge(h));
                    let fill_corners = border
                        .and_then(|h| mesh.grid_fill_corners(h).first().copied())
                        .filter(|_| b % 2 == 0);
                    let face_pair = face.and_then(|f| {
                        let corners = mesh.face_corners(f);
                        (corners.len() > 3).then(|| (f, corners[0].vertex, corners[2].vertex))
                    });
                    let result = match kind {
                        0 => edge.map(|h| doc.edit("Split edge", |m| m.split_edge(h, t).map(|_| ()))),
                        1 => face_pair.map(|(f, x, y)| doc.edit("Split face", |m| m.split_face(f, x, y).map(|_| ()))),
                        2 => vertex.zip(other).map(|(v, w)| {
                            doc.edit("Merge", |m| m.merge_vertices(&[v, w], Vec3::ZERO).map(|_| ()))
                        }),
                        3 => vertex.map(|v| doc.edit("Dissolve vertex", |m| m.dissolve_vertex(v))),
                        4 => edge.map(|h| doc.edit("Dissolve edge", |m| m.dissolve_edges(&[h.edge()]))),
                        5 => edge.map(|h| doc.edit("Rotate edge", |m| m.rotate_edge(h.edge(), b % 2 == 0).map(|_| ()))),
                        6 => edge.map(|h| doc.edit("Insert loop", |m| m.insert_loop(h, t).map(|_| ()))),
                        8 => border.map(|h| {
                            doc.edit("Extrude", |m| {
                                m.extrude_border(&[h], |v| Vec3::new(v.0 as f32, 0.0, 1.0)).map(|_| ())
                            })
                        }),
                        9 => border.map(|h| doc.edit("Fill", |m| m.fill_hole(h, fill_corners).map(|_| ()))),
                        10 => edge.map(|h| {
                            doc.edit("Toggle hard", |m| {
                                let hard = !m.is_hard(h.edge());
                                m.set_hard(h.edge(), hard);
                                Ok(())
                            })
                        }),
                        _ => face.zip(second).map(|(f, g)| {
                            doc.edit("Dissolve faces", |m| m.dissolve_faces(&[f, g]).map(|_| ()))
                        }),
                    };
                    result.unwrap_or(Ok(()))
                }
                Op::Undo => { doc.undo(); Ok(()) }
                Op::Redo => { doc.redo(); Ok(()) }
            };
            prop_assert!(doc.mesh().validate().is_ok(), "{:?}", doc.mesh().validate());
            match (&op, result) {
                (Op::Undo | Op::Redo, _) => {}
                (_, Err(_)) => prop_assert_eq!(doc.mesh(), &before, "failed edit changed the mesh"),
                (_, Ok(())) if doc.undo_steps() > steps => {
                    let after = doc.mesh().clone();
                    doc.undo();
                    prop_assert_eq!(doc.mesh(), &before, "undo did not restore the previous state");
                    doc.redo();
                    prop_assert_eq!(doc.mesh(), &after, "redo did not restore the edit");
                }
                (_, Ok(())) => prop_assert_eq!(doc.mesh(), &before, "edit without an undo step changed the mesh"),
            }
        }
        while doc.undo().is_some() {}
        prop_assert_eq!(doc.mesh(), &initial);
    }
}

/// Found by the property test: a face that closes one fan of a vertex into a
/// disk must be refused while other fans touch that vertex.
#[test]
fn closing_a_fan_shared_with_other_fans_is_refused() {
    let build = |extra_fans: bool| {
        let mut mesh = HalfEdgeMesh::new();
        let p = |m: &mut HalfEdgeMesh, x: f32, y: f32| m.add_vertex(Vec3::new(x, y, 0.0));
        let c = p(&mut mesh, 0.0, 0.0);
        let (x, y, z) = (
            p(&mut mesh, 1.0, 0.0),
            p(&mut mesh, 1.0, 1.0),
            p(&mut mesh, 0.0, 1.0),
        );
        mesh.add_face(&[c, x, y, z]).unwrap();
        if extra_fans {
            let (a, b) = (p(&mut mesh, -1.0, 0.5), p(&mut mesh, -1.0, -0.5));
            mesh.add_face(&[c, a, b]).unwrap();
            let (d, e) = (p(&mut mesh, 0.5, -1.0), p(&mut mesh, -0.5, -1.0));
            mesh.add_face(&[c, d, e]).unwrap();
        }
        mesh.validate().unwrap();
        (mesh, [c, x, y, z])
    };
    // Alone, the quad's fan may be closed (topologically valid, if folded).
    // The triangle reuses both quad edges at `c`, reversed: x -> c and c -> z.
    let (mut mesh, [c, x, _, z]) = build(false);
    mesh.add_face(&[x, c, z]).unwrap();
    mesh.validate().unwrap();

    let (mut mesh, [c, x, _, z]) = build(true);
    let before = mesh.clone();
    assert_eq!(
        mesh.add_face(&[x, c, z]),
        Err(TopologyError::PatchRelinkFailed)
    );
    assert_eq!(mesh, before);
}

/// Timing for a large retopology mesh; run with
/// `cargo test --release -p polyflow-core large_mesh_timing -- --ignored --nocapture`.
#[test]
fn uvs_belong_to_face_corners() {
    use glam::Vec2;
    let (mesh, cells) = grid(2);
    let mut doc = MeshDocument::new(mesh);
    let mut faces = Vec::new();
    for cell in &cells {
        faces.push(doc.edit("Add face", |m| m.add_face(cell)).unwrap());
    }
    assert!(!doc.mesh().has_uvs());
    assert_eq!(doc.mesh().face_uvs(faces[0]), None);

    // A seam: the shared vertex gets different UVs in each face.
    let corners = [
        Vec2::new(0.0, 0.0),
        Vec2::new(0.5, 0.0),
        Vec2::new(0.5, 0.5),
        Vec2::new(0.0, 0.5),
    ];
    let shifted = corners.map(|uv| uv + Vec2::new(0.6, 0.0));
    doc.edit("UVs", |m| {
        m.set_face_uvs(faces[0], &corners);
        m.set_face_uvs(faces[1], &shifted);
        Ok::<_, TopologyError>(())
    })
    .unwrap();
    let mesh = doc.mesh();
    mesh.validate().unwrap();
    assert!(mesh.has_uvs());
    assert_eq!(mesh.face_uvs(faces[0]).unwrap(), corners);
    assert_eq!(mesh.face_uvs(faces[1]).unwrap(), shifted);
    assert_eq!(mesh.face_uvs(faces[2]), None, "untouched faces have no UVs");
    // Boundary half-edges have no corner, so setting a UV there does nothing.
    let boundary = mesh
        .halfedges()
        .find(|&h| mesh.is_boundary_halfedge(h))
        .unwrap();
    doc.edit("Boundary UV", |m| {
        m.set_uv(boundary, Some(Vec2::ONE));
        Ok::<_, TopologyError>(())
    })
    .unwrap();
    assert_eq!(doc.mesh().uv(boundary), None);

    // Removing a face drops its corners' UVs, and undo brings them back.
    doc.edit("Remove", |m| m.remove_face(faces[0], false))
        .unwrap();
    doc.mesh().validate().unwrap();
    doc.undo();
    assert_eq!(doc.mesh().face_uvs(faces[0]).unwrap(), corners);
    doc.undo();
    assert!(!doc.mesh().has_uvs());
    doc.redo();
    assert_eq!(doc.mesh().face_uvs(faces[1]).unwrap(), shifted);
    doc.edit("Clear", |m| {
        m.clear_uvs();
        Ok::<_, TopologyError>(())
    })
    .unwrap();
    assert!(!doc.mesh().has_uvs());
    doc.mesh().validate().unwrap();
}

#[test]
#[ignore]
fn large_mesh_timing() {
    use std::time::Instant;
    let n = 300;
    let (mesh, cells) = grid(n);
    let mut doc = MeshDocument::new(mesh);
    doc.set_max_steps(usize::MAX);
    let start = Instant::now();
    for cell in &cells {
        doc.edit("Add face", |m| m.add_face(cell)).unwrap();
    }
    let add = start.elapsed();
    let start = Instant::now();
    while doc.undo().is_some() {}
    let undo = start.elapsed();
    let start = Instant::now();
    while doc.redo().is_some() {}
    let redo = start.elapsed();
    let start = Instant::now();
    doc.mesh().validate().unwrap();
    let validate = start.elapsed();
    let faces = cells.len() as f64;
    println!(
        "{} quads: add {:.2} us/face, undo {:.2} us/step, redo {:.2} us/step, validate {:?}",
        cells.len(),
        add.as_secs_f64() / faces * 1e6,
        undo.as_secs_f64() / faces * 1e6,
        redo.as_secs_f64() / faces * 1e6,
        validate
    );
}

#[test]
fn loops_and_rings_on_a_grid() {
    let n = 4;
    let (mut mesh, cells) = grid(n);
    add_all(&mut mesh, &cells);
    let at = |i: usize, j: usize| VertexId((j * (n + 1) + i) as u32);
    let edge = |a, b| mesh.find_halfedge(a, b).unwrap().edge();
    let ends = |edges: &[EdgeId]| -> Vec<VertexId> {
        let mut v: Vec<VertexId> = edges
            .iter()
            .flat_map(|e| [mesh.origin(e.halfedge(0)), mesh.origin(e.halfedge(1))])
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };

    // A row of horizontal edges, from border to border, in order.
    let e = edge(at(1, 2), at(2, 2));
    let row = mesh.edge_loop(e);
    assert_eq!(row.len(), 4);
    assert_eq!(ends(&row), (0..=n).map(|i| at(i, 2)).collect::<Vec<_>>());
    assert!(
        row.windows(2).all(|w| ends(w).len() == 3),
        "consecutive edges touch"
    );

    // Across the quads of column 1: the 5 horizontal edges above each other.
    let ring = mesh.edge_ring(e);
    assert_eq!(ring.len(), n + 1);
    assert!(ring.contains(&edge(at(1, 0), at(2, 0))));
    assert!(ring.contains(&edge(at(1, n), at(2, n))));
    let faces = mesh.face_loop(e);
    assert_eq!(faces.len(), n);
    for f in faces {
        assert!(
            mesh.face_vertices(f)
                .any(|v| v == at(1, 0) || v == at(1, 1) || v == at(1, 2) || v == at(1, 3))
        );
    }

    // A border edge gives the whole border.
    let border = mesh.edge_loop(edge(at(0, 0), at(1, 0)));
    assert_eq!(border.len(), 4 * n);
    assert!(border.iter().all(|&e| mesh.is_boundary_edge(e)));
}

#[test]
fn rings_close_around_a_cube() {
    let corners = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ];
    let quads = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    let mut mesh = HalfEdgeMesh::new();
    let v: Vec<VertexId> = corners
        .iter()
        .map(|c| mesh.add_vertex(Vec3::from_array(c.map(|x| x as f32))))
        .collect();
    add_all(&mut mesh, &quads.map(|q| q.map(|i| v[i])));
    let e = mesh.find_halfedge(v[0], v[1]).unwrap().edge();
    let ring = mesh.edge_ring(e);
    assert_eq!(ring.len(), 4);
    assert_eq!(ring[0], e, "a closed ring starts at its edge");
    assert_eq!(mesh.face_loop(e).len(), 4);
    // Cube corners have three edges: a loop cannot continue through them.
    assert_eq!(mesh.edge_loop(e), vec![e]);
    assert_eq!(mesh.valence(v[0]), 3);
}

/// A grid whose corners carry UVs equal to their XY position.
fn uv_grid(n: usize) -> HalfEdgeMesh {
    let (mut mesh, cells) = grid(n);
    add_all(&mut mesh, &cells);
    let faces: Vec<FaceId> = mesh.faces().collect();
    for f in faces {
        let uvs: Vec<Vec2> = mesh
            .face_vertices(f)
            .map(|v| mesh.position(v).truncate())
            .collect();
        mesh.set_face_uvs(f, &uvs);
    }
    mesh
}

/// Every corner's UV is its vertex's XY position.
fn uvs_follow_positions(mesh: &HalfEdgeMesh) {
    for f in mesh.faces() {
        for h in mesh.face_halfedges(f) {
            let uv = mesh.uv(h).expect("corner keeps a UV");
            let p = mesh.position(mesh.origin(h)).truncate();
            assert!(uv.distance(p) < 1e-5, "{uv} at {p}");
        }
    }
}

fn at(n: usize, i: usize, j: usize) -> VertexId {
    VertexId((j * (n + 1) + i) as u32)
}

#[test]
fn split_edges_and_faces() {
    let n = 2;
    let mut mesh = uv_grid(n);
    let h = mesh.find_halfedge(at(n, 1, 0), at(n, 1, 1)).unwrap();
    let m = mesh.split_edge(h, 0.25).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.position(m), Vec3::new(1.0, 0.25, 0.0));
    assert_eq!(mesh.vertex_faces(m).count(), 2);
    assert!(
        mesh.vertex_faces(m)
            .all(|f| mesh.face_vertices(f).count() == 5)
    );
    uvs_follow_positions(&mesh);

    let f = mesh.vertex_faces(at(n, 0, 0)).next().unwrap();
    mesh.split_face(f, at(n, 0, 0), m).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 5);
    uvs_follow_positions(&mesh);
    assert!(matches!(
        mesh.split_face(f, at(n, 0, 0), at(n, 1, 0)),
        Err(TopologyError::MissingFace(_) | TopologyError::Refused(_))
    ));
}

#[test]
fn merge_vertices_collapses_faces() {
    let n = 2;
    let mut mesh = uv_grid(n);
    let (a, b) = (at(n, 1, 1), at(n, 2, 1));
    let kept = mesh
        .merge_vertices(&[a, b], Vec3::new(1.5, 1.0, 0.0))
        .unwrap();
    mesh.validate().unwrap();
    assert_eq!(kept, a);
    assert!(!mesh.contains_vertex(b));
    assert_eq!(mesh.vertex_count(), 8);
    assert_eq!(mesh.face_count(), 4, "two quads became triangles");
    assert_eq!(mesh.position(a), Vec3::new(1.5, 1.0, 0.0));
}

#[test]
fn dissolving() {
    let n = 4;
    let mut mesh = uv_grid(n);
    // A loop across the middle: faces merge in pairs and the loop's inner
    // vertices, left between two edges, go too.
    let e = mesh.find_halfedge(at(n, 1, 2), at(n, 2, 2)).unwrap().edge();
    let row = mesh.edge_loop(e);
    mesh.dissolve_edges(&row).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 12);
    assert_eq!(mesh.vertex_count(), 22);
    // Border vertices stay: the two merged faces at the ends keep them.
    let sides = |k| {
        mesh.faces()
            .filter(|&f| mesh.face_vertices(f).count() == k)
            .count()
    };
    assert_eq!((sides(4), sides(5)), (10, 2));
    uvs_follow_positions(&mesh);

    let mut mesh = uv_grid(n);
    let v = at(n, 2, 2);
    mesh.dissolve_vertex(v).unwrap();
    mesh.validate().unwrap();
    assert!(!mesh.contains_vertex(v));
    assert_eq!(mesh.face_count(), 13);
    assert!(mesh.faces().any(|f| mesh.face_vertices(f).count() == 8));
    uvs_follow_positions(&mesh);

    // A ring of faces around a hole has no single outline: refused, and
    // the document rolls back.
    let mut doc = MeshDocument::new(uv_grid(3));
    let before = doc.mesh().clone();
    let center = doc.mesh().vertex_faces(at(3, 1, 1)).collect::<HashSet<_>>();
    let center = *center
        .iter()
        .find(|&&f| doc.mesh().vertex_faces(at(3, 2, 2)).any(|g| g == f))
        .unwrap();
    let ring: Vec<FaceId> = doc.mesh().faces().filter(|&f| f != center).collect();
    assert!(doc.edit("Dissolve", |m| m.dissolve_faces(&ring)).is_err());
    assert_eq!(doc.mesh(), &before);
}

#[test]
fn rotate_edges() {
    let n = 2;
    let mut mesh = uv_grid(n);
    let e = mesh.find_halfedge(at(n, 1, 0), at(n, 1, 1)).unwrap().edge();
    let turned = mesh.rotate_edge(e, true).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 4);
    assert!(mesh.faces().all(|f| mesh.face_vertices(f).count() == 4));
    let ends = [turned.halfedge(0), turned.halfedge(1)].map(|h| mesh.origin(h));
    assert!(!ends.contains(&at(n, 1, 0)) || !ends.contains(&at(n, 1, 1)));
    uvs_follow_positions(&mesh);
    let border = mesh.find_halfedge(at(n, 0, 0), at(n, 1, 0)).unwrap().edge();
    assert!(mesh.rotate_edge(border, true).is_err());
    // Turning back restores the original edge.
    let back = mesh.rotate_edge(turned, false).unwrap();
    assert!(
        mesh.find_halfedge(at(n, 1, 0), at(n, 1, 1))
            .is_some_and(|h| h.edge() == back)
    );
}

#[test]
fn insert_loops() {
    let n = 4;
    let mut mesh = uv_grid(n);
    let h = mesh.find_halfedge(at(n, 1, 2), at(n, 2, 2)).unwrap();
    let new = mesh.insert_loop(h, 0.5).unwrap();
    mesh.validate().unwrap();
    assert_eq!(new.len(), n + 1);
    assert_eq!(mesh.face_count(), n * n + n);
    assert!(new.iter().all(|&v| mesh.position(v).x == 1.5));
    assert!(mesh.faces().all(|f| mesh.face_vertices(f).count() == 4));
    uvs_follow_positions(&mesh);

    // An open ring ending at a triangle: the triangle gets the vertex.
    let n = 2;
    let (mut mesh, cells) = grid(n);
    add_all(&mut mesh, &cells[..3]);
    let tri = [at(n, 1, 1), at(n, 2, 1), at(n, 2, 2)];
    mesh.add_face(&tri).unwrap();
    let h = mesh.find_halfedge(at(n, 1, 0), at(n, 2, 0)).unwrap();
    mesh.insert_loop(h, 0.3).unwrap();
    mesh.validate().unwrap();
    assert!(
        mesh.faces().any(|f| mesh.face_vertices(f).count() == 4
            && mesh.face_vertices(f).any(|v| v == at(n, 2, 2)))
    );

    // A closed ring around a cube.
    let corners = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ];
    let quads = [
        [0, 3, 2, 1],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ];
    let mut mesh = HalfEdgeMesh::new();
    let v: Vec<VertexId> = corners
        .iter()
        .map(|c| mesh.add_vertex(Vec3::from_array(c.map(|x| x as f32))))
        .collect();
    add_all(&mut mesh, &quads.map(|q| q.map(|i| v[i])));
    let h = mesh.find_halfedge(v[0], v[1]).unwrap();
    assert_eq!(mesh.insert_loop(h, 0.5).unwrap().len(), 4);
    mesh.validate().unwrap();
    assert_eq!((mesh.vertex_count(), mesh.face_count()), (12, 10));
}

#[test]
fn extrude_and_bridge_borders() {
    let n = 2;
    let (mut mesh, cells) = grid(n);
    add_all(&mut mesh, &cells);
    // The bottom border, pushed down.
    let bottom: Vec<EdgeId> = (0..n)
        .map(|i| {
            mesh.find_halfedge(at(n, i, 0), at(n, i + 1, 0))
                .unwrap()
                .edge()
        })
        .collect();
    let chains = mesh.border_chains(&bottom);
    assert_eq!(chains.len(), 1);
    assert_eq!(chains[0].len(), 2);
    assert!(!mesh.is_closed_chain(&chains[0]));
    let copies = mesh
        .extrude_border(&chains[0], mesh_position_below)
        .unwrap();
    mesh.validate().unwrap();
    assert_eq!(copies.len(), 3);
    assert_eq!((mesh.vertex_count(), mesh.face_count()), (12, 6));

    // Two separate rows of quads, bridged across the gap between them.
    let (mut mesh, cells) = grid(3);
    let rows: Vec<[VertexId; 4]> = cells[..3].iter().chain(&cells[6..]).copied().collect();
    add_all(&mut mesh, &rows);
    let lower: Vec<EdgeId> = (0..3)
        .map(|i| {
            mesh.find_halfedge(at(3, i, 1), at(3, i + 1, 1))
                .unwrap()
                .edge()
        })
        .collect();
    let upper: Vec<EdgeId> = (0..3)
        .map(|i| {
            mesh.find_halfedge(at(3, i, 2), at(3, i + 1, 2))
                .unwrap()
                .edge()
        })
        .collect();
    let (a, b) = (mesh.border_chains(&lower), mesh.border_chains(&upper));
    mesh.bridge(&a[0], &b[0]).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 9);
    assert_eq!(boundary_vertices(&mesh), 12, "one border all around");
    // Unequal borders are refused.
    let (mut mesh, cells) = grid(3);
    let partial: Vec<[VertexId; 4]> = cells[..3].iter().chain(&cells[6..8]).copied().collect();
    add_all(&mut mesh, &partial);
    let row_edges = |mesh: &HalfEdgeMesh, j: usize, count: usize| -> Vec<EdgeId> {
        (0..count)
            .map(|i| {
                mesh.find_halfedge(at(3, i, j), at(3, i + 1, j))
                    .unwrap()
                    .edge()
            })
            .collect()
    };
    let (a, b) = (
        mesh.border_chains(&row_edges(&mesh, 1, 3)),
        mesh.border_chains(&row_edges(&mesh, 2, 2)),
    );
    assert!(mesh.bridge(&a[0], &b[0]).is_err());
}

fn mesh_position_below(v: VertexId) -> Vec3 {
    Vec3::new(v.0 as f32, -1.0, 0.0)
}

#[test]
fn closed_borders_bridge_and_holes_fill() {
    // A 4x4 grid without its middle 2x2: a hole of 8 edges.
    let n = 4;
    let (mut mesh, cells) = grid(n);
    let outer: Vec<[VertexId; 4]> = cells
        .iter()
        .enumerate()
        .filter(|(k, _)| !matches!(k, 5 | 6 | 9 | 10))
        .map(|(_, c)| *c)
        .collect();
    add_all(&mut mesh, &outer);
    mesh.remove_vertex(at(n, 2, 2)).unwrap();
    let h = mesh.find_halfedge(at(n, 1, 1), at(n, 2, 1)).unwrap();
    assert!(mesh.is_boundary_halfedge(h));
    let corners = mesh.grid_fill_corners(h);
    let best = corners[0];
    let mut expected = [at(n, 1, 1), at(n, 3, 1), at(n, 3, 3), at(n, 1, 3)];
    let mut got = best;
    expected.sort_unstable();
    got.sort_unstable();
    assert_eq!(got, expected, "corners where the border turns");
    let inner = mesh.fill_hole(h, Some(best)).unwrap();
    mesh.validate().unwrap();
    assert_eq!(inner.len(), 1);
    assert!((mesh.position(inner[0]) - Vec3::new(2.0, 2.0, 0.0)).length() < 1e-5);
    assert_eq!(mesh.face_count(), 16);
    assert_eq!(boundary_vertices(&mesh), 16);

    // Odd holes take one polygon.
    let (mut mesh, _) = grid(1);
    let v: Vec<VertexId> = mesh.vertices().collect();
    mesh.add_face(&[v[0], v[1], v[3]]).unwrap();
    let h = mesh.find_halfedge(v[1], v[0]).unwrap();
    assert!(mesh.grid_fill_corners(h).is_empty());
    mesh.fill_hole(h, None).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 2);

    // Two closed borders: the ends of an open tube, bridged into a closed
    // ring. Squares at z = 0 and z = 1, with a strip of quads on z = 0..1
    // missing: bridging the strip's borders closes the tube.
    let mut mesh = HalfEdgeMesh::new();
    let ring = |mesh: &mut HalfEdgeMesh, z: f32| -> Vec<VertexId> {
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .map(|&(x, y)| mesh.add_vertex(Vec3::new(x, y, z)))
            .collect()
    };
    let (r0, r1, r2) = (
        ring(&mut mesh, 0.0),
        ring(&mut mesh, 1.0),
        ring(&mut mesh, 2.0),
    );
    for k in 0..4 {
        let k1 = (k + 1) % 4;
        mesh.add_face(&[r0[k], r0[k1], r1[k1], r1[k]]).unwrap();
    }
    // A separate band further along, facing the same way.
    let r3 = ring(&mut mesh, 3.0);
    for k in 0..4 {
        let k1 = (k + 1) % 4;
        mesh.add_face(&[r2[k], r2[k1], r3[k1], r3[k]]).unwrap();
    }
    let edges = |r: &[VertexId]| -> Vec<EdgeId> {
        (0..4)
            .map(|k| mesh.find_halfedge(r[k], r[(k + 1) % 4]).unwrap().edge())
            .collect()
    };
    let (a, b) = (
        mesh.border_chains(&edges(&r1)),
        mesh.border_chains(&edges(&r2)),
    );
    assert!(mesh.is_closed_chain(&a[0]) && mesh.is_closed_chain(&b[0]));
    mesh.bridge(&a[0], &b[0]).unwrap();
    mesh.validate().unwrap();
    assert_eq!(mesh.face_count(), 12);
    // Lined up with matching corners: bridge edges are all straight.
    for k in 0..4 {
        assert!(mesh.find_halfedge(r1[k], r2[k]).is_some());
    }
}

#[test]
fn hard_edges_are_undoable_and_survive_topology_edits() {
    let n = 3;
    let mut doc = MeshDocument::new(uv_grid(n));
    let edge = |m: &HalfEdgeMesh, a, b| m.find_halfedge(a, b).unwrap().edge();
    // The middle column of vertical edges, x = 1.
    let column: Vec<(VertexId, VertexId)> =
        (0..n).map(|j| (at(n, 1, j), at(n, 1, j + 1))).collect();
    doc.edit("Mark hard", |m| {
        for &(a, b) in &column {
            let e = edge(m, a, b);
            m.set_hard(e, true);
        }
        Ok::<_, TopologyError>(())
    })
    .unwrap();
    assert_eq!(doc.mesh().hard_edges().count(), 3);
    doc.undo();
    assert!(!doc.mesh().has_hard_edges());
    doc.redo();
    assert_eq!(doc.mesh().hard_edges().count(), 3);

    // A loop across the column (through a ring of vertical edges) splits
    // one of its edges in two.
    let h = doc.mesh().find_halfedge(at(n, 1, 0), at(n, 1, 1)).unwrap();
    doc.edit("Insert loop", |m| m.insert_loop(h, 0.5)).unwrap();
    doc.mesh().validate().unwrap();
    assert_eq!(
        doc.mesh().hard_edges().count(),
        4,
        "the split edge's halves stay hard"
    );
    doc.undo();

    // Splitting a hard edge keeps both halves hard; dissolving the new
    // vertex joins them back into one hard edge.
    let h = doc.mesh().find_halfedge(at(n, 1, 0), at(n, 1, 1)).unwrap();
    let m = doc.edit("Split", |mesh| mesh.split_edge(h, 0.5)).unwrap();
    assert!(doc.mesh().is_hard(edge(doc.mesh(), at(n, 1, 0), m)));
    assert!(doc.mesh().is_hard(edge(doc.mesh(), m, at(n, 1, 1))));
    doc.edit("Dissolve", |mesh| mesh.dissolve_vertex(m))
        .unwrap();
    assert!(
        doc.mesh()
            .is_hard(edge(doc.mesh(), at(n, 1, 0), at(n, 1, 1)))
    );

    // Rebuilt faces keep the hard edges they share with others.
    let f = doc.mesh().find_halfedge(at(n, 0, 0), at(n, 1, 0)).unwrap();
    let face = doc.mesh().face_of(f).unwrap();
    let corners = doc.mesh().face_corners(face);
    doc.edit("Split face", |mesh| {
        mesh.split_face(face, corners[0].vertex, corners[2].vertex)
            .map(|_| ())
    })
    .unwrap();
    assert!(
        doc.mesh()
            .is_hard(edge(doc.mesh(), at(n, 1, 0), at(n, 1, 1)))
    );
    doc.mesh().validate().unwrap();
}
