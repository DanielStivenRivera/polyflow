//! The selection: vertices, edges or faces of the retopology, depending on
//! the selection mode.
//!
//! Selection is not part of the undo history. It holds element ids, which
//! stay valid across edits and undo; ids of removed elements are dropped by
//! [`Selection::prune`].

use std::collections::HashSet;

use polyflow_core::halfedge::{EdgeId, FaceId, HalfEdgeMesh, VertexId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectMode {
    #[default]
    Vertex,
    Edge,
    Face,
}

impl SelectMode {
    pub const ALL: [SelectMode; 3] = [SelectMode::Vertex, SelectMode::Edge, SelectMode::Face];

    pub fn label(self) -> &'static str {
        match self {
            SelectMode::Vertex => "Vertices",
            SelectMode::Edge => "Edges",
            SelectMode::Face => "Faces",
        }
    }

    pub fn shortcut(self) -> &'static str {
        match self {
            SelectMode::Vertex => "1",
            SelectMode::Edge => "2",
            SelectMode::Face => "3",
        }
    }
}

/// How a pick combines with the current selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectOp {
    Replace,
    Add,
    Remove,
}

/// Elements of one kind, as picked or as selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Elements {
    Vertices(Vec<VertexId>),
    Edges(Vec<EdgeId>),
    Faces(Vec<FaceId>),
}

/// What the selected edges and faces were made of, to tell after an edit
/// whether an id still names the same element (see [`Selection::revalidate`]).
#[derive(Debug, Clone, Default)]
pub struct Snapshot {
    edges: Vec<(EdgeId, [VertexId; 2])>,
    faces: Vec<(FaceId, Vec<VertexId>)>,
}

#[derive(Debug, Clone, Default)]
pub struct Selection {
    mode: SelectMode,
    /// Only the set of the current mode is used; the others stay empty.
    vertices: HashSet<VertexId>,
    edges: HashSet<EdgeId>,
    faces: HashSet<FaceId>,
    /// Bumped on every change, so views can tell when to refresh.
    generation: u64,
}

impl Selection {
    pub fn mode(&self) -> SelectMode {
        self.mode
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn len(&self) -> usize {
        match self.mode {
            SelectMode::Vertex => self.vertices.len(),
            SelectMode::Edge => self.edges.len(),
            SelectMode::Face => self.faces.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn vertices(&self) -> &HashSet<VertexId> {
        &self.vertices
    }

    pub fn edges(&self) -> &HashSet<EdgeId> {
        &self.edges
    }

    pub fn faces(&self) -> &HashSet<FaceId> {
        &self.faces
    }

    pub fn clear(&mut self) {
        if !self.is_empty() {
            self.vertices.clear();
            self.edges.clear();
            self.faces.clear();
            self.generation += 1;
        }
    }

    /// Switches mode, converting the selection: to a smaller element, the
    /// parts of what was selected; to a larger one, the elements whose parts
    /// were all selected.
    pub fn set_mode(&mut self, mesh: &HalfEdgeMesh, mode: SelectMode) {
        if mode == self.mode {
            return;
        }
        let vertices = std::mem::take(&mut self.vertices);
        let edges = std::mem::take(&mut self.edges);
        let faces = std::mem::take(&mut self.faces);
        let edge_ends = |e: EdgeId| [mesh.origin(e.halfedge(0)), mesh.origin(e.halfedge(1))];
        match (self.mode, mode) {
            (SelectMode::Vertex, SelectMode::Edge) => {
                self.edges = mesh
                    .edges()
                    .filter(|&e| edge_ends(e).iter().all(|v| vertices.contains(v)))
                    .collect();
            }
            (SelectMode::Vertex, SelectMode::Face) => {
                self.faces = mesh
                    .faces()
                    .filter(|&f| mesh.face_vertices(f).all(|v| vertices.contains(&v)))
                    .collect();
            }
            (SelectMode::Edge, SelectMode::Vertex) => {
                self.vertices = edges.iter().flat_map(|&e| edge_ends(e)).collect();
            }
            (SelectMode::Edge, SelectMode::Face) => {
                self.faces = mesh
                    .faces()
                    .filter(|&f| mesh.face_halfedges(f).all(|h| edges.contains(&h.edge())))
                    .collect();
            }
            (SelectMode::Face, SelectMode::Vertex) => {
                self.vertices = faces
                    .iter()
                    .flat_map(|&f| mesh.face_vertices(f).collect::<Vec<_>>())
                    .collect();
            }
            (SelectMode::Face, SelectMode::Edge) => {
                self.edges = faces
                    .iter()
                    .flat_map(|&f| mesh.face_halfedges(f).map(|h| h.edge()).collect::<Vec<_>>())
                    .collect();
            }
            _ => unreachable!("modes differ"),
        }
        self.mode = mode;
        self.generation += 1;
    }

    /// Combines picked elements with the selection. Elements of another kind
    /// than the current mode are ignored.
    pub fn apply(&mut self, op: SelectOp, elements: Elements) {
        if op == SelectOp::Replace {
            self.clear();
        }
        fn combine<T: std::hash::Hash + Eq>(
            set: &mut HashSet<T>,
            op: SelectOp,
            items: Vec<T>,
        ) -> bool {
            let mut changed = false;
            for item in items {
                changed |= match op {
                    SelectOp::Remove => set.remove(&item),
                    SelectOp::Replace | SelectOp::Add => set.insert(item),
                };
            }
            changed
        }
        let changed = match (self.mode, elements) {
            (SelectMode::Vertex, Elements::Vertices(v)) => combine(&mut self.vertices, op, v),
            (SelectMode::Edge, Elements::Edges(e)) => combine(&mut self.edges, op, e),
            (SelectMode::Face, Elements::Faces(f)) => combine(&mut self.faces, op, f),
            _ => false,
        };
        if changed {
            self.generation += 1;
        }
    }

    /// Selects every element of the current mode.
    pub fn select_all(&mut self, mesh: &HalfEdgeMesh) {
        let all = match self.mode {
            SelectMode::Vertex => Elements::Vertices(mesh.vertices().collect()),
            SelectMode::Edge => Elements::Edges(mesh.edges().collect()),
            SelectMode::Face => Elements::Faces(mesh.faces().collect()),
        };
        self.apply(SelectOp::Add, all);
    }

    /// Selects what was not selected and deselects the rest.
    pub fn invert(&mut self, mesh: &HalfEdgeMesh) {
        match self.mode {
            SelectMode::Vertex => {
                self.vertices = mesh
                    .vertices()
                    .filter(|v| !self.vertices.contains(v))
                    .collect()
            }
            SelectMode::Edge => {
                self.edges = mesh.edges().filter(|e| !self.edges.contains(e)).collect()
            }
            SelectMode::Face => {
                self.faces = mesh.faces().filter(|f| !self.faces.contains(f)).collect()
            }
        }
        self.generation += 1;
    }

    /// Records what the selected edges and faces are made of.
    pub fn snapshot(&self, mesh: &HalfEdgeMesh) -> Snapshot {
        Snapshot {
            edges: self.edges.iter().map(|&e| (e, edge_key(mesh, e))).collect(),
            faces: self.faces.iter().map(|&f| (f, face_key(mesh, f))).collect(),
        }
    }

    /// Drops edges and faces that were removed or rebuilt since `snapshot`:
    /// an edit can free an element's slot and give it to a new element, which
    /// must not inherit the selection.
    pub fn revalidate(&mut self, mesh: &HalfEdgeMesh, snapshot: &Snapshot) {
        self.prune(mesh);
        let before = self.len();
        for (e, key) in &snapshot.edges {
            if self.edges.contains(e) && edge_key(mesh, *e) != *key {
                self.edges.remove(e);
            }
        }
        for (f, key) in &snapshot.faces {
            if self.faces.contains(f) && face_key(mesh, *f) != *key {
                self.faces.remove(f);
            }
        }
        if self.len() != before {
            self.generation += 1;
        }
    }

    /// Drops removed elements. Call whenever the mesh changed.
    pub fn prune(&mut self, mesh: &HalfEdgeMesh) {
        let before = self.len();
        self.vertices.retain(|&v| mesh.contains_vertex(v));
        self.edges
            .retain(|&e| mesh.contains_halfedge(e.halfedge(0)));
        self.faces.retain(|&f| mesh.contains_face(f));
        if self.len() != before {
            self.generation += 1;
        }
    }

    /// The selected edges, or in vertex mode the edges between selected
    /// vertices, as loops and rings grow from edges.
    pub fn edges_for_loops(&self, mesh: &HalfEdgeMesh) -> Vec<EdgeId> {
        match self.mode {
            SelectMode::Vertex => mesh
                .edges()
                .filter(|&e| {
                    self.vertices.contains(&mesh.origin(e.halfedge(0)))
                        && self.vertices.contains(&mesh.origin(e.halfedge(1)))
                })
                .collect(),
            SelectMode::Edge => self.edges.iter().copied().collect(),
            // Edges between two selected faces: the face loop through them
            // crosses those edges.
            SelectMode::Face => mesh
                .edges()
                .filter(|&e| {
                    [e.halfedge(0), e.halfedge(1)]
                        .iter()
                        .all(|&h| mesh.face_of(h).is_some_and(|f| self.faces.contains(&f)))
                })
                .collect(),
        }
    }

    /// Adds the loop through each of `edges` (see [`loop_elements`]).
    pub fn grow_loops(&mut self, mesh: &HalfEdgeMesh, edges: &[EdgeId], ring: bool) {
        for &e in edges {
            let elements = loop_elements(mesh, self.mode, e, ring);
            self.apply(SelectOp::Add, elements);
        }
    }
}

fn edge_key(mesh: &HalfEdgeMesh, e: EdgeId) -> [VertexId; 2] {
    let mut ends = [mesh.origin(e.halfedge(0)), mesh.origin(e.halfedge(1))];
    ends.sort_unstable();
    ends
}

fn face_key(mesh: &HalfEdgeMesh, f: FaceId) -> Vec<VertexId> {
    let mut vertices: Vec<VertexId> = mesh.face_vertices(f).collect();
    vertices.sort_unstable();
    vertices
}

/// The loop through `e` as elements of `mode`: in vertex and edge modes, the
/// edge loop (or ring); in face mode, the face loop crossing `e`.
pub fn loop_elements(mesh: &HalfEdgeMesh, mode: SelectMode, e: EdgeId, ring: bool) -> Elements {
    let edges = || {
        if ring {
            mesh.edge_ring(e)
        } else {
            mesh.edge_loop(e)
        }
    };
    match mode {
        SelectMode::Vertex => {
            let mut vertices: Vec<VertexId> = edges()
                .into_iter()
                .flat_map(|e| [mesh.origin(e.halfedge(0)), mesh.origin(e.halfedge(1))])
                .collect();
            vertices.sort_unstable();
            vertices.dedup();
            Elements::Vertices(vertices)
        }
        SelectMode::Edge => Elements::Edges(edges()),
        SelectMode::Face => Elements::Faces(mesh.face_loop(e)),
    }
}

#[cfg(test)]
mod tests {
    use polyflow_core::glam::Vec3;

    use super::*;

    /// A 3x3 grid of quads on the XY plane.
    fn grid() -> (HalfEdgeMesh, Vec<VertexId>) {
        let mut mesh = HalfEdgeMesh::new();
        let ids: Vec<VertexId> = (0..16)
            .map(|k| mesh.add_vertex(Vec3::new((k % 4) as f32, (k / 4) as f32, 0.0)))
            .collect();
        for j in 0..3 {
            for i in 0..3 {
                let k = j * 4 + i;
                mesh.add_face(&[ids[k], ids[k + 1], ids[k + 5], ids[k + 4]])
                    .unwrap();
            }
        }
        (mesh, ids)
    }

    #[test]
    fn modes_convert() {
        let (mesh, ids) = grid();
        let mut sel = Selection::default();
        // The four corners of the middle quad.
        sel.apply(
            SelectOp::Replace,
            Elements::Vertices(vec![ids[5], ids[6], ids[9], ids[10]]),
        );
        sel.set_mode(&mesh, SelectMode::Edge);
        assert_eq!(sel.len(), 4);
        sel.set_mode(&mesh, SelectMode::Face);
        assert_eq!(sel.len(), 1);
        sel.set_mode(&mesh, SelectMode::Vertex);
        assert_eq!(sel.len(), 4);
        sel.set_mode(&mesh, SelectMode::Face);
        sel.set_mode(&mesh, SelectMode::Edge);
        assert_eq!(sel.len(), 4);
    }

    #[test]
    fn add_remove_invert_and_prune() {
        let (mut mesh, ids) = grid();
        let mut sel = Selection::default();
        sel.apply(SelectOp::Add, Elements::Vertices(vec![ids[0], ids[1]]));
        sel.apply(SelectOp::Remove, Elements::Vertices(vec![ids[0]]));
        assert_eq!(sel.vertices().iter().copied().collect::<Vec<_>>(), [ids[1]]);
        sel.apply(SelectOp::Add, Elements::Faces(mesh.faces().collect()));
        assert_eq!(sel.len(), 1, "other kinds are ignored");
        sel.invert(&mesh);
        assert_eq!(sel.len(), 15);
        let generation = sel.generation();
        mesh.remove_vertex(ids[15]).unwrap();
        sel.prune(&mesh);
        assert_eq!(sel.len(), 14);
        assert_ne!(sel.generation(), generation);
    }

    #[test]
    fn loops_from_the_selection() {
        let (mesh, ids) = grid();
        let mut sel = Selection::default();
        sel.set_mode(&mesh, SelectMode::Face);
        // Two faces side by side in the bottom row: the loop is the row.
        let faces: Vec<FaceId> = mesh.vertex_faces(ids[1]).collect();
        assert_eq!(faces.len(), 2);
        sel.apply(SelectOp::Replace, Elements::Faces(faces));
        let edges = sel.edges_for_loops(&mesh);
        assert_eq!(edges.len(), 1);
        sel.grow_loops(&mesh, &edges, false);
        assert_eq!(sel.len(), 3);

        sel.set_mode(&mesh, SelectMode::Vertex);
        sel.clear();
        sel.apply(SelectOp::Add, Elements::Vertices(vec![ids[5], ids[6]]));
        let edges = sel.edges_for_loops(&mesh);
        sel.grow_loops(&mesh, &edges, false);
        assert_eq!(sel.len(), 4, "the row of vertices through the middle");
    }
}
