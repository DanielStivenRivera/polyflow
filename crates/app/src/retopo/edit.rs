//! Retopology edits that honor symmetry. Run them inside
//! `MeshDocument::edit` so each user action is one atomic undo step.

use polyflow_core::glam::Vec3;
use polyflow_core::halfedge::{FaceId, HalfEdgeMesh, TopologyError, VertexId};

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
    let mirrored = symmetry.and_then(|sym| {
        let vertices: Vec<VertexId> = mesh.face_vertices(f).collect();
        let partners: Option<Vec<VertexId>> =
            vertices.iter().map(|&v| sym.partner(mesh, v)).collect();
        let partners = partners?;
        mesh.vertex_faces(partners[0])
            .find(|&g| g != f && same_set(&mesh.face_vertices(g).collect::<Vec<_>>(), &partners))
    });
    mesh.remove_face(f, true)?;
    if let Some(g) = mirrored {
        mesh.remove_face(g, true)?;
    }
    Ok(())
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
