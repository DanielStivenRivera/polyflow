//! Retopology edits that honor symmetry. Run them inside
//! `MeshDocument::edit` so each user action is one atomic undo step.

use std::collections::HashSet;

use polyflow_core::glam::Vec3;
use polyflow_core::halfedge::{EdgeId, FaceId, HalfEdgeId, HalfEdgeMesh, TopologyError, VertexId};

use super::selection::{SelectMode, Selection};
use super::symmetry::Symmetry;

/// Adds a vertex (and its mirror partner). Positions near the mirror plane
/// snap onto it and make a single seam vertex; a position already holding a
/// vertex (e.g. a partner created earlier) reuses it.
pub fn add_vertex(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    position: Vec3,
) -> VertexId {
    let Some(sym) = symmetry else {
        return mesh.add_vertex(position);
    };
    let position = sym.snap_to_seam(position);
    if let Some(existing) = sym.vertex_at(mesh, position) {
        return existing;
    }
    let v = mesh.add_vertex(position);
    sym.partner_or_create(mesh, v);
    v
}

/// Moves a vertex (and its partner). Seam vertices stay on the plane.
pub fn move_vertex(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    v: VertexId,
    position: Vec3,
) {
    let Some(sym) = symmetry else {
        mesh.set_position(v, position);
        return;
    };
    match sym.partner(mesh, v) {
        Some(partner) if partner == v => {
            let mut p = position;
            p[sym.axis.index()] = 0.0;
            mesh.set_position(v, p);
        }
        Some(partner) => {
            mesh.set_position(v, position);
            mesh.set_position(partner, sym.reflect(position));
        }
        None => mesh.set_position(v, position),
    }
}

/// Newell's normal of a polygon (not normalized; zero if degenerate).
pub fn polygon_normal(positions: &[Vec3]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for (i, a) in positions.iter().enumerate() {
        let b = positions[(i + 1) % positions.len()];
        n += Vec3::new(
            (a.y - b.y) * (a.z + b.z),
            (a.z - b.z) * (a.x + b.x),
            (a.x - b.x) * (a.y + b.y),
        );
    }
    n
}

/// Adds a face through `vertices`, ordered so it faces `facing` (e.g. the
/// camera or the surface normal). If the edges around it demand the other
/// orientation, that one is used instead.
pub fn add_face_facing(
    mesh: &mut HalfEdgeMesh,
    vertices: &[VertexId],
    facing: Vec3,
) -> Result<FaceId, TopologyError> {
    let positions: Vec<Vec3> = vertices.iter().map(|&v| mesh.position(v)).collect();
    let mut ordered = vertices.to_vec();
    if polygon_normal(&positions).dot(facing) < 0.0 {
        ordered.reverse();
    }
    match mesh.add_face(&ordered) {
        Err(TopologyError::EdgeTaken(..)) => {
            ordered.reverse();
            mesh.add_face(&ordered)
        }
        result => result,
    }
}

/// Adds a face facing `facing`, and its mirror image.
pub fn add_face(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    vertices: &[VertexId],
    facing: Vec3,
) -> Result<FaceId, TopologyError> {
    let face = add_face_facing(mesh, vertices, facing)?;
    if let Some(sym) = symmetry {
        let mirrored: Vec<VertexId> = vertices
            .iter()
            .map(|&v| sym.partner_or_create(mesh, v))
            .collect();
        // A face that maps onto itself straddles the seam: no copy needed.
        if !same_set(&mirrored, vertices) {
            add_face_facing(mesh, &mirrored, sym.reflect(facing))?;
        }
    }
    Ok(face)
}

/// Removes a vertex (and its partner) with their faces. Neighbors left
/// without any face are removed too, so no stray vertices remain.
pub fn remove_vertex(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    v: VertexId,
) -> Result<(), TopologyError> {
    let partner = symmetry
        .and_then(|sym| sym.partner(mesh, v))
        .filter(|&p| p != v);
    remove_vertex_and_strays(mesh, v)?;
    if let Some(p) = partner.filter(|&p| mesh.contains_vertex(p)) {
        remove_vertex_and_strays(mesh, p)?;
    }
    Ok(())
}

/// Removes a face (and its mirror image). Vertices left alone are removed.
pub fn remove_face(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    f: FaceId,
) -> Result<(), TopologyError> {
    let mirrored = symmetry
        .and_then(|sym| mirror_face(mesh, sym, f))
        .filter(|&g| g != f);
    mesh.remove_face(f, true)?;
    if let Some(g) = mirrored {
        mesh.remove_face(g, true)?;
    }
    Ok(())
}

/// The mirror image of a face, if there is one (itself if it straddles the
/// seam).
pub fn mirror_face(mesh: &HalfEdgeMesh, sym: &Symmetry, f: FaceId) -> Option<FaceId> {
    let partners: Option<Vec<VertexId>> = mesh
        .face_vertices(f)
        .map(|v| sym.partner(mesh, v))
        .collect();
    let partners = partners?;
    mesh.vertex_faces(partners[0])
        .find(|&g| same_set(&mesh.face_vertices(g).collect::<Vec<_>>(), &partners))
}

/// The mirror image of a half-edge: from the partner of its origin to the
/// partner of its target.
pub fn mirror_halfedge(mesh: &HalfEdgeMesh, sym: &Symmetry, h: HalfEdgeId) -> Option<HalfEdgeId> {
    let a = sym.partner(mesh, mesh.origin(h))?;
    let b = sym.partner(mesh, mesh.target(h))?;
    mesh.find_halfedge(a, b)
}

/// Selected edges and faces with their mirror images, and selected vertices
/// with their partners.
fn with_mirrors(
    mesh: &HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    selection: &Selection,
) -> Selected {
    let mut vertices: Vec<VertexId> = selection.vertices().iter().copied().collect();
    let mut edges: Vec<EdgeId> = selection.edges().iter().copied().collect();
    let mut faces: Vec<FaceId> = selection.faces().iter().copied().collect();
    if let Some(sym) = symmetry {
        let mirrored: Vec<VertexId> = vertices
            .iter()
            .filter_map(|&v| sym.partner(mesh, v))
            .collect();
        vertices.extend(mirrored);
        let mirrored: Vec<EdgeId> = edges
            .iter()
            .filter_map(|&e| mirror_halfedge(mesh, sym, e.halfedge(0)).map(|h| h.edge()))
            .collect();
        edges.extend(mirrored);
        let mirrored: Vec<FaceId> = faces
            .iter()
            .filter_map(|&f| mirror_face(mesh, sym, f))
            .collect();
        faces.extend(mirrored);
    }
    vertices.sort_unstable();
    vertices.dedup();
    edges.sort_unstable();
    edges.dedup();
    faces.sort_unstable();
    faces.dedup();
    Selected {
        vertices,
        edges,
        faces,
    }
}

struct Selected {
    vertices: Vec<VertexId>,
    edges: Vec<EdgeId>,
    faces: Vec<FaceId>,
}

/// Deletes the selected elements (and their mirror images): vertices with
/// their faces, edges with the faces along them, faces. Vertices left
/// without faces go too.
pub fn delete_selection(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    selection: &Selection,
) -> Result<(), TopologyError> {
    let selected = with_mirrors(mesh, symmetry, selection);
    match selection.mode() {
        SelectMode::Vertex => {
            for v in selected.vertices {
                if mesh.contains_vertex(v) {
                    remove_vertex_and_strays(mesh, v)?;
                }
            }
        }
        SelectMode::Edge | SelectMode::Face => {
            let mut faces: Vec<FaceId> = selected
                .edges
                .iter()
                .flat_map(|e| [e.halfedge(0), e.halfedge(1)])
                .filter_map(|h| mesh.face_of(h))
                .chain(selected.faces)
                .collect();
            faces.sort_unstable();
            faces.dedup();
            for f in faces {
                mesh.remove_face(f, true)?;
            }
        }
    }
    Ok(())
}

/// Dissolves the selection (and its mirror image): vertices merge the faces
/// around them, edges the faces on both sides, faces merge into one polygon
/// per connected group.
pub fn dissolve_selection(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    selection: &Selection,
) -> Result<(), TopologyError> {
    let selected = with_mirrors(mesh, symmetry, selection);
    match selection.mode() {
        SelectMode::Vertex => {
            for v in selected.vertices {
                if mesh.contains_vertex(v) {
                    mesh.dissolve_vertex(v)?;
                }
            }
            Ok(())
        }
        SelectMode::Edge => mesh.dissolve_edges(&selected.edges),
        SelectMode::Face => mesh.dissolve_faces(&selected.faces).map(|_| ()),
    }
}

/// Merges vertices into one at `position`, and their partners into one at
/// its reflection. Vertices that span the mirror plane (some of them are
/// each other's partners, or on the seam) merge into a single seam vertex.
pub fn merge_vertices(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    vertices: &[VertexId],
    position: Vec3,
) -> Result<VertexId, TopologyError> {
    let Some(sym) = symmetry else {
        return mesh.merge_vertices(vertices, position);
    };
    let set: HashSet<VertexId> = vertices.iter().copied().collect();
    let partners: Vec<Option<VertexId>> = vertices.iter().map(|&v| sym.partner(mesh, v)).collect();
    let spans = partners.iter().flatten().any(|p| set.contains(p));
    if spans {
        let mut all: Vec<VertexId> = vertices.to_vec();
        all.extend(partners.iter().flatten().filter(|p| !set.contains(p)));
        let mut p = position;
        p[sym.axis.index()] = 0.0;
        return mesh.merge_vertices(&all, p);
    }
    let kept = mesh.merge_vertices(vertices, position)?;
    let partners: Option<Vec<VertexId>> = partners.into_iter().collect();
    if let Some(partners) = partners {
        mesh.merge_vertices(&partners, sym.reflect(position))?;
    }
    Ok(kept)
}

/// Rotates edges (see `HalfEdgeMesh::rotate_edge`); mirror images turn the
/// mirrored way. Edges that are their own mirror image are left alone with
/// symmetry on, since turning them would break it.
pub fn rotate_edges(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    edges: &[EdgeId],
) -> Result<Vec<EdgeId>, TopologyError> {
    // Pair edges with their mirrors before turning any: pairing is by
    // endpoints, which turning changes.
    let mut jobs: Vec<(EdgeId, bool)> = Vec::new();
    let mut done: HashSet<EdgeId> = HashSet::new();
    for &e in edges {
        if !done.insert(e) {
            continue;
        }
        match symmetry.map(|sym| mirror_halfedge(mesh, sym, e.halfedge(0)).map(|h| h.edge())) {
            Some(Some(m)) if m == e => continue,
            Some(Some(m)) => {
                jobs.push((e, true));
                if done.insert(m) {
                    jobs.push((m, false));
                }
            }
            _ => jobs.push((e, true)),
        }
    }
    let mut turned = Vec::new();
    for (e, ccw) in jobs {
        turned.push(mesh.rotate_edge(e, ccw)?);
    }
    Ok(turned)
}

/// The face holding both `a` and `b` where they are not already neighbors.
fn face_between(mesh: &HalfEdgeMesh, a: VertexId, b: VertexId) -> Option<FaceId> {
    mesh.vertex_faces(a).find(|&f| {
        let vertices: Vec<VertexId> = mesh.face_vertices(f).collect();
        let n = vertices.len();
        let (Some(ia), Some(ib)) = (
            vertices.iter().position(|&v| v == a),
            vertices.iter().position(|&v| v == b),
        ) else {
            return false;
        };
        (ia + 1) % n != ib && (ib + 1) % n != ia
    })
}

/// Splits the face shared by `a` and `b` along a new edge between them (and
/// the mirror face between their partners). Returns the new edge.
pub fn connect(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    a: VertexId,
    b: VertexId,
) -> Result<EdgeId, TopologyError> {
    let Some(f) = face_between(mesh, a, b) else {
        return Err(TopologyError::Refused(
            "the vertices do not share a face, or are already joined by an edge",
        ));
    };
    let mirrored = symmetry.and_then(|sym| Some((sym.partner(mesh, a)?, sym.partner(mesh, b)?)));
    let e = mesh.split_face(f, a, b)?;
    if let Some((pa, pb)) = mirrored
        && !same_set(&[pa, pb], &[a, b])
        && let Some(g) = face_between(mesh, pa, pb)
    {
        mesh.split_face(g, pa, pb)?;
    }
    Ok(e)
}

/// Splits `h`'s edge at `t` (and its mirror image at the mirrored point).
/// An edge crossing the mirror plane is split on the plane instead.
pub fn split_edge(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    h: HalfEdgeId,
    t: f32,
) -> Result<VertexId, TopologyError> {
    let Some(sym) = symmetry else {
        return mesh.split_edge(h, t);
    };
    match mirror_halfedge(mesh, sym, h) {
        Some(m) if m.edge() == h.edge() => {
            let axis = sym.axis.index();
            let (a, b) = (
                mesh.position(mesh.origin(h))[axis],
                mesh.position(mesh.target(h))[axis],
            );
            let t = if (a - b).abs() > 1e-12 {
                a / (a - b)
            } else {
                t
            };
            let v = mesh.split_edge(h, t.clamp(0.0, 1.0))?;
            let mut p = mesh.position(v);
            p[axis] = 0.0;
            mesh.set_position(v, p);
            Ok(v)
        }
        Some(m) => {
            let v = mesh.split_edge(h, t)?;
            mesh.split_edge(m, t)?;
            Ok(v)
        }
        None => mesh.split_edge(h, t),
    }
}

/// Where an edge loop through `h` at `t` goes with symmetry: the half-edge
/// and fraction to insert at, and the mirror ring's, if it is another ring.
/// A ring that is its own mirror image the other way round must be cut in
/// the middle to stay symmetric.
pub fn loop_plan(
    mesh: &HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    h: HalfEdgeId,
    t: f32,
) -> (f32, Option<HalfEdgeId>) {
    let Some(m) = symmetry.and_then(|sym| mirror_halfedge(mesh, sym, h)) else {
        return (t, None);
    };
    let (ring, _) = mesh.ring_halfedges(h);
    match ring.iter().find(|r| r.edge() == m.edge()) {
        Some(&r) if r == m => (t, None),
        Some(_) => (0.5, None),
        None => (t, Some(m)),
    }
}

/// Inserts an edge loop through `h` at `t` (see `HalfEdgeMesh::insert_loop`)
/// and its mirror image. Returns the new vertices.
pub fn insert_loop(
    mesh: &mut HalfEdgeMesh,
    symmetry: Option<&Symmetry>,
    h: HalfEdgeId,
    t: f32,
) -> Result<Vec<VertexId>, TopologyError> {
    let (t, mirror) = loop_plan(mesh, symmetry, h, t);
    // The mirror ring's edge by its ends: inserting the first loop may
    // rebuild faces, but keeps the ends' vertices.
    let mirror_ends = mirror.map(|m| (mesh.origin(m), mesh.target(m)));
    let mut new = mesh.insert_loop(h, t)?;
    if let Some((a, b)) = mirror_ends
        && let Some(m) = mesh.find_halfedge(a, b)
    {
        new.extend(mesh.insert_loop(m, t)?);
    }
    Ok(new)
}

fn remove_vertex_and_strays(mesh: &mut HalfEdgeMesh, v: VertexId) -> Result<(), TopologyError> {
    // Every vertex of the surrounding faces, not just edge neighbors: the
    // opposite corner of a quad can be left alone too.
    let mut neighbors: Vec<VertexId> = mesh
        .vertex_faces(v)
        .flat_map(|f| mesh.face_vertices(f).collect::<Vec<_>>())
        .filter(|&n| n != v)
        .collect();
    neighbors.sort_unstable();
    neighbors.dedup();
    mesh.remove_vertex(v)?;
    for n in neighbors {
        if mesh.outgoing(n).is_none() {
            mesh.remove_vertex(n)?;
        }
    }
    Ok(())
}

fn same_set(a: &[VertexId], b: &[VertexId]) -> bool {
    a.len() == b.len() && a.iter().all(|v| b.contains(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retopo::symmetry::Axis;

    #[test]
    fn faces_are_oriented_and_mirrored() {
        let sym = Symmetry::new(Axis::X, 1.0);
        let mut mesh = HalfEdgeMesh::new();
        // Clockwise when seen from +Z; facing +Z must flip it.
        let quad = [(0.2, 0.0), (0.2, 1.0), (1.0, 1.0), (1.0, 0.0)]
            .map(|(x, y)| add_vertex(&mut mesh, Some(&sym), Vec3::new(x, y, 0.0)));
        assert_eq!(mesh.vertex_count(), 8, "every vertex got a partner");
        let f = add_face(&mut mesh, Some(&sym), &quad, Vec3::Z).unwrap();
        let positions: Vec<Vec3> = mesh.face_vertices(f).map(|v| mesh.position(v)).collect();
        assert!(polygon_normal(&positions).z > 0.0);
        assert_eq!(mesh.face_count(), 2);
        for g in mesh.faces() {
            let positions: Vec<Vec3> = mesh.face_vertices(g).map(|v| mesh.position(v)).collect();
            assert!(
                polygon_normal(&positions).z > 0.0,
                "the mirror also faces +Z"
            );
        }
        mesh.validate().unwrap();

        remove_face(&mut mesh, Some(&sym), f).unwrap();
        assert_eq!((mesh.face_count(), mesh.vertex_count()), (0, 0));

        // Deleting one corner removes both quads and leaves no stray vertices.
        let quad = [(0.2, 0.0), (0.2, 1.0), (1.0, 1.0), (1.0, 0.0)]
            .map(|(x, y)| add_vertex(&mut mesh, Some(&sym), Vec3::new(x, y, 0.0)));
        add_face(&mut mesh, Some(&sym), &quad, Vec3::Z).unwrap();
        remove_vertex(&mut mesh, Some(&sym), quad[0]).unwrap();
        assert_eq!((mesh.face_count(), mesh.vertex_count()), (0, 0));
        mesh.validate().unwrap();
    }

    #[test]
    fn seam_vertices_stay_single_and_on_the_plane() {
        let sym = Symmetry::new(Axis::X, 1.0);
        let mut mesh = HalfEdgeMesh::new();
        let s = add_vertex(&mut mesh, Some(&sym), Vec3::new(0.002, 0.5, 0.0));
        assert_eq!(mesh.vertex_count(), 1);
        assert_eq!(mesh.position(s).x, 0.0);
        move_vertex(&mut mesh, Some(&sym), s, Vec3::new(0.3, 0.6, 0.0));
        assert_eq!(mesh.position(s), Vec3::new(0.0, 0.6, 0.0));

        let a = add_vertex(&mut mesh, Some(&sym), Vec3::new(0.5, 0.0, 0.0));
        move_vertex(&mut mesh, Some(&sym), a, Vec3::new(0.7, 0.1, 0.0));
        assert!(
            sym.vertex_at(&mesh, Vec3::new(-0.7, 0.1, 0.0)).is_some(),
            "partner followed"
        );
    }

    /// A 4x2 grid of quads across the X mirror plane, facing +Z.
    fn mirrored_grid(sym: &Symmetry) -> (HalfEdgeMesh, Vec<VertexId>) {
        let mut mesh = HalfEdgeMesh::new();
        let ids: Vec<VertexId> = (0..15)
            .map(|k| {
                let (i, j) = ((k % 5) as f32, (k / 5) as f32);
                add_vertex(&mut mesh, Some(sym), Vec3::new(i - 2.0, j, 0.0))
            })
            .collect();
        for j in 0..2 {
            for i in 0..4 {
                let k = j * 5 + i;
                add_face_facing(
                    &mut mesh,
                    &[ids[k], ids[k + 1], ids[k + 6], ids[k + 5]],
                    Vec3::Z,
                )
                .unwrap();
            }
        }
        (mesh, ids)
    }

    fn is_symmetric(mesh: &HalfEdgeMesh, sym: &Symmetry) -> bool {
        mesh.vertices().all(|v| sym.partner(mesh, v).is_some())
            && mesh.faces().all(|f| mirror_face(mesh, sym, f).is_some())
    }

    #[test]
    fn symmetric_topology_edits() {
        let sym = Symmetry::new(Axis::X, 2.0);
        // An edge loop on one side gets its mirror loop.
        let (mut mesh, ids) = mirrored_grid(&sym);
        let h = mesh.find_halfedge(ids[0], ids[1]).unwrap();
        let new = insert_loop(&mut mesh, Some(&sym), h, 0.3).unwrap();
        mesh.validate().unwrap();
        assert_eq!(new.len(), 6);
        assert!(is_symmetric(&mesh, &sym));

        // Splitting an edge also splits its mirror; connecting mirrors too.
        let (mut mesh, ids) = mirrored_grid(&sym);
        let h = mesh.find_halfedge(ids[0], ids[5]).unwrap();
        let v = split_edge(&mut mesh, Some(&sym), h, 0.5).unwrap();
        assert!(is_symmetric(&mesh, &sym));
        connect(&mut mesh, Some(&sym), v, ids[6]).unwrap();
        mesh.validate().unwrap();
        assert!(is_symmetric(&mesh, &sym));
        assert_eq!(mesh.face_count(), 10);

        // Rotating an edge turns its mirror the mirrored way.
        let (mut mesh, ids) = mirrored_grid(&sym);
        let e = mesh.find_halfedge(ids[1], ids[6]).unwrap().edge();
        rotate_edges(&mut mesh, Some(&sym), &[e]).unwrap();
        mesh.validate().unwrap();
        assert!(is_symmetric(&mesh, &sym));

        // Merging two vertices merges their partners; merging partners
        // with each other makes a seam vertex.
        let (mut mesh, ids) = mirrored_grid(&sym);
        merge_vertices(
            &mut mesh,
            Some(&sym),
            &[ids[10], ids[11]],
            Vec3::new(-1.5, 2.0, 0.0),
        )
        .unwrap();
        mesh.validate().unwrap();
        assert!(is_symmetric(&mesh, &sym));
        assert_eq!(mesh.vertex_count(), 13);
        let seam = merge_vertices(&mut mesh, Some(&sym), &[ids[6]], Vec3::new(-1.0, 1.0, 0.0));
        assert!(seam.is_ok());
    }

    #[test]
    fn deleting_and_dissolving_the_selection() {
        use crate::retopo::selection::{Elements, SelectOp};

        let sym = Symmetry::new(Axis::X, 2.0);
        let (mut mesh, ids) = mirrored_grid(&sym);
        let mut selection = Selection::default();
        selection.set_mode(&mesh, SelectMode::Face);
        let corner = mesh.vertex_faces(ids[0]).next().unwrap();
        selection.apply(SelectOp::Replace, Elements::Faces(vec![corner]));
        delete_selection(&mut mesh, Some(&sym), &selection).unwrap();
        assert_eq!(mesh.face_count(), 6, "the face and its mirror");
        assert!(is_symmetric(&mesh, &sym));

        let (mut mesh, ids) = mirrored_grid(&sym);
        selection.set_mode(&mesh, SelectMode::Edge);
        let e = mesh.find_halfedge(ids[6], ids[7]).unwrap().edge();
        selection.apply(SelectOp::Replace, Elements::Edges(vec![e]));
        dissolve_selection(&mut mesh, Some(&sym), &selection).unwrap();
        mesh.validate().unwrap();
        assert_eq!(mesh.face_count(), 6, "two pairs of quads merged");
        assert!(is_symmetric(&mesh, &sym));
    }

    #[test]
    fn faces_across_the_seam_are_not_duplicated() {
        let sym = Symmetry::new(Axis::X, 1.0);
        let mut mesh = HalfEdgeMesh::new();
        let v = [(-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (-0.5, 1.0)]
            .map(|(x, y)| add_vertex(&mut mesh, Some(&sym), Vec3::new(x, y, 0.0)));
        let unique: std::collections::HashSet<_> = v.iter().collect();
        assert_eq!(unique.len(), 4);
        add_face(&mut mesh, Some(&sym), &v, Vec3::Z).unwrap();
        assert_eq!(mesh.face_count(), 1);
    }
}
