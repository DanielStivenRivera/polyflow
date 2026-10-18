//! Editable polygon mesh for retopology, as a half-edge structure.
//!
//! Elements live in slot arrays and are addressed by stable ids: removing an
//! element frees its slot without moving others, so ids held by tools and
//! selections stay valid, and undo brings back the very same ids.
//!
//! Every write goes through [`HalfEdgeMesh::set_vertex`],
//! [`HalfEdgeMesh::set_halfedge`] or [`HalfEdgeMesh::set_face`], which record
//! the old and new value of the element in a journal while one is open. That
//! journal is what makes any operation built on top undoable (see
//! [`crate::history`]), with no per-operation undo code.
//!
//! Invariants (checked by [`HalfEdgeMesh::validate`]):
//! - Half-edges come in pairs: the twin of `h` is `h ^ 1`, and both are live
//!   or both free. A half-edge with no face is a boundary half-edge; at least
//!   one half of every edge has a face (there are no loose edges).
//! - `next`/`prev` form closed loops: around each face, and around each hole.
//! - Rotating around a vertex (`next(twin(h))`) visits all its outgoing
//!   half-edges. Several fans may meet at a boundary vertex (two quads touching
//!   at a corner while a patch is being built); they are chained through their
//!   boundary gaps. Interior vertices have exactly one fan.
//! - If a vertex is on a boundary, its `outgoing` half-edge is a boundary one.

mod ops;
#[cfg(test)]
mod tests;
mod validate;

use glam::Vec3;

pub use ops::TopologyError;

macro_rules! id_type {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);

        impl $name {
            #[inline]
            pub fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_type!(
    /// A vertex of a [`HalfEdgeMesh`].
    VertexId
);
id_type!(
    /// A directed half of an edge.
    HalfEdgeId
);
id_type!(
    /// A polygon of a [`HalfEdgeMesh`].
    FaceId
);
id_type!(
    /// An undirected edge: half-edges `2e` and `2e + 1`.
    EdgeId
);

impl HalfEdgeId {
    #[inline]
    pub fn twin(self) -> Self {
        Self(self.0 ^ 1)
    }

    #[inline]
    pub fn edge(self) -> EdgeId {
        EdgeId(self.0 / 2)
    }
}

impl EdgeId {
    #[inline]
    pub fn halfedge(self, side: u32) -> HalfEdgeId {
        HalfEdgeId(self.0 * 2 + (side & 1))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Vertex {
    pub position: Vec3,
    /// An outgoing half-edge (a boundary one if any), or `None` if isolated.
    pub outgoing: Option<HalfEdgeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HalfEdge {
    pub origin: VertexId,
    pub next: HalfEdgeId,
    pub prev: HalfEdgeId,
    /// `None` on boundaries.
    pub face: Option<FaceId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Face {
    pub halfedge: HalfEdgeId,
}

/// One recorded write: the element's value before and after.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Change {
    Vertex(u32, Option<Vertex>, Option<Vertex>),
    HalfEdge(u32, Option<HalfEdge>, Option<HalfEdge>),
    Face(u32, Option<Face>, Option<Face>),
}

/// A polygon mesh with half-edge connectivity. See the [module docs](self).
#[derive(Debug, Clone, Default)]
pub struct HalfEdgeMesh {
    vertices: Vec<Option<Vertex>>,
    halfedges: Vec<Option<HalfEdge>>,
    faces: Vec<Option<Face>>,
    /// Candidate free slots; may hold stale entries (slots refilled by undo),
    /// which allocation skips.
    free_vertices: Vec<u32>,
    free_edges: Vec<u32>,
    free_faces: Vec<u32>,
    vertex_count: usize,
    edge_count: usize,
    face_count: usize,
    journal: Option<Vec<Change>>,
    /// Incremented on every change, so views can tell when to refresh.
    version: u64,
}

impl PartialEq for HalfEdgeMesh {
    /// Meshes are equal when their elements are, slot by slot (free-list and
    /// journal bookkeeping is ignored).
    fn eq(&self, other: &Self) -> bool {
        fn trimmed<T>(slots: &[Option<T>]) -> &[Option<T>] {
            let len = slots.iter().rposition(Option::is_some).map_or(0, |i| i + 1);
            &slots[..len]
        }
        trimmed(&self.vertices) == trimmed(&other.vertices)
            && trimmed(&self.halfedges) == trimmed(&other.halfedges)
            && trimmed(&self.faces) == trimmed(&other.faces)
    }
}

impl HalfEdgeMesh {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn face_count(&self) -> usize {
        self.face_count
    }

    pub fn is_empty(&self) -> bool {
        self.vertex_count == 0
    }

    /// Changes whenever the mesh does (including through undo and redo).
    pub fn version(&self) -> u64 {
        self.version
    }

    // ---- Element access -------------------------------------------------

    pub fn contains_vertex(&self, v: VertexId) -> bool {
        matches!(self.vertices.get(v.index()), Some(Some(_)))
    }

    pub fn contains_face(&self, f: FaceId) -> bool {
        matches!(self.faces.get(f.index()), Some(Some(_)))
    }

    pub fn contains_halfedge(&self, h: HalfEdgeId) -> bool {
        matches!(self.halfedges.get(h.index()), Some(Some(_)))
    }

    pub(crate) fn vertex(&self, v: VertexId) -> &Vertex {
        self.vertices[v.index()].as_ref().expect("live vertex")
    }

    pub(crate) fn he(&self, h: HalfEdgeId) -> &HalfEdge {
        self.halfedges[h.index()].as_ref().expect("live half-edge")
    }

    pub(crate) fn face(&self, f: FaceId) -> &Face {
        self.faces[f.index()].as_ref().expect("live face")
    }

    pub fn position(&self, v: VertexId) -> Vec3 {
        self.vertex(v).position
    }

    pub fn outgoing(&self, v: VertexId) -> Option<HalfEdgeId> {
        self.vertex(v).outgoing
    }

    pub fn origin(&self, h: HalfEdgeId) -> VertexId {
        self.he(h).origin
    }

    pub fn target(&self, h: HalfEdgeId) -> VertexId {
        self.he(h.twin()).origin
    }

    pub fn next(&self, h: HalfEdgeId) -> HalfEdgeId {
        self.he(h).next
    }

    pub fn prev(&self, h: HalfEdgeId) -> HalfEdgeId {
        self.he(h).prev
    }

    pub fn face_of(&self, h: HalfEdgeId) -> Option<FaceId> {
        self.he(h).face
    }

    pub fn face_halfedge(&self, f: FaceId) -> HalfEdgeId {
        self.face(f).halfedge
    }

    pub fn is_boundary_halfedge(&self, h: HalfEdgeId) -> bool {
        self.he(h).face.is_none()
    }

    pub fn is_boundary_edge(&self, e: EdgeId) -> bool {
        self.is_boundary_halfedge(e.halfedge(0)) || self.is_boundary_halfedge(e.halfedge(1))
    }

    /// True for isolated vertices and vertices on a boundary.
    pub fn is_boundary_vertex(&self, v: VertexId) -> bool {
        self.vertex(v)
            .outgoing
            .is_none_or(|h| self.is_boundary_halfedge(h))
    }

    // ---- Iteration ------------------------------------------------------

    pub fn vertices(&self) -> impl Iterator<Item = VertexId> + '_ {
        self.vertices
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_some())
            .map(|(i, _)| VertexId(i as u32))
    }

    pub fn faces(&self) -> impl Iterator<Item = FaceId> + '_ {
        self.faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.is_some())
            .map(|(i, _)| FaceId(i as u32))
    }

    pub fn edges(&self) -> impl Iterator<Item = EdgeId> + '_ {
        self.halfedges
            .iter()
            .step_by(2)
            .enumerate()
            .filter(|(_, h)| h.is_some())
            .map(|(i, _)| EdgeId(i as u32))
    }

    pub fn halfedges(&self) -> impl Iterator<Item = HalfEdgeId> + '_ {
        self.halfedges
            .iter()
            .enumerate()
            .filter(|(_, h)| h.is_some())
            .map(|(i, _)| HalfEdgeId(i as u32))
    }

    /// Half-edges of a face, in order.
    pub fn face_halfedges(&self, f: FaceId) -> impl Iterator<Item = HalfEdgeId> + '_ {
        self.loop_from(self.face(f).halfedge)
    }

    /// Vertices of a face, counter-clockwise.
    pub fn face_vertices(&self, f: FaceId) -> impl Iterator<Item = VertexId> + '_ {
        self.face_halfedges(f).map(|h| self.origin(h))
    }

    /// The `next` loop starting at `start` (a face or a hole).
    pub fn loop_from(&self, start: HalfEdgeId) -> impl Iterator<Item = HalfEdgeId> + '_ {
        let mut current = Some(start);
        std::iter::from_fn(move || {
            let h = current?;
            let next = self.next(h);
            current = (next != start).then_some(next);
            Some(h)
        })
    }

    /// Half-edges leaving `v`, rotating around it.
    pub fn outgoing_halfedges(&self, v: VertexId) -> impl Iterator<Item = HalfEdgeId> + '_ {
        let start = self.vertex(v).outgoing;
        let mut current = start;
        std::iter::from_fn(move || {
            let h = current?;
            let next = self.next(h.twin());
            current = (Some(next) != start).then_some(next);
            Some(h)
        })
    }

    /// Vertices sharing an edge with `v`.
    pub fn vertex_neighbors(&self, v: VertexId) -> impl Iterator<Item = VertexId> + '_ {
        self.outgoing_halfedges(v).map(|h| self.target(h))
    }

    /// Faces around `v`.
    pub fn vertex_faces(&self, v: VertexId) -> impl Iterator<Item = FaceId> + '_ {
        self.outgoing_halfedges(v).filter_map(|h| self.face_of(h))
    }

    /// The half-edge from `from` to `to`, if they share an edge.
    pub fn find_halfedge(&self, from: VertexId, to: VertexId) -> Option<HalfEdgeId> {
        self.outgoing_halfedges(from)
            .find(|&h| self.target(h) == to)
    }

    /// Fan triangulation of every face, as indices into the vertex slots
    /// (suitable for drawing alongside [`HalfEdgeMesh::vertex_slots`]).
    pub fn triangulate(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.face_count * 6);
        for f in self.faces() {
            let mut verts = self.face_vertices(f);
            let Some(first) = verts.next() else { continue };
            let mut prev = verts.next();
            for v in verts {
                if let Some(p) = prev {
                    out.extend_from_slice(&[first.0, p.0, v.0]);
                }
                prev = Some(v);
            }
        }
        out
    }

    /// Positions indexed by vertex slot; free slots hold `None`.
    pub fn vertex_slots(&self) -> impl Iterator<Item = Option<Vec3>> + '_ {
        self.vertices.iter().map(|v| v.map(|v| v.position))
    }

    // ---- Writes (all journaled) ----------------------------------------

    pub(crate) fn set_vertex(&mut self, v: VertexId, value: Option<Vertex>) {
        let slot = grow(&mut self.vertices, v.index());
        let old = std::mem::replace(slot, value);
        if old == value {
            return;
        }
        self.vertex_count = self.vertex_count + value.is_some() as usize - old.is_some() as usize;
        if value.is_none() {
            self.free_vertices.push(v.0);
        }
        self.version += 1;
        if let Some(journal) = &mut self.journal {
            journal.push(Change::Vertex(v.0, old, value));
        }
    }

    pub(crate) fn set_halfedge(&mut self, h: HalfEdgeId, value: Option<HalfEdge>) {
        // Grow to cover both halves of the edge.
        grow(&mut self.halfedges, h.index() | 1);
        let slot = &mut self.halfedges[h.index()];
        let old = std::mem::replace(slot, value);
        if old == value {
            return;
        }
        // Edge bookkeeping happens on the even half, which is written last
        // when creating an edge and first when removing one (see `ops`).
        if h.0.is_multiple_of(2) {
            self.edge_count = self.edge_count + value.is_some() as usize - old.is_some() as usize;
            if value.is_none() {
                self.free_edges.push(h.0 / 2);
            }
        }
        self.version += 1;
        if let Some(journal) = &mut self.journal {
            journal.push(Change::HalfEdge(h.0, old, value));
        }
    }

    pub(crate) fn set_face(&mut self, f: FaceId, value: Option<Face>) {
        let slot = grow(&mut self.faces, f.index());
        let old = std::mem::replace(slot, value);
        if old == value {
            return;
        }
        self.face_count = self.face_count + value.is_some() as usize - old.is_some() as usize;
        if value.is_none() {
            self.free_faces.push(f.0);
        }
        self.version += 1;
        if let Some(journal) = &mut self.journal {
            journal.push(Change::Face(f.0, old, value));
        }
    }

    /// Moves a vertex.
    pub fn set_position(&mut self, v: VertexId, position: Vec3) {
        let vertex = Vertex {
            position,
            ..*self.vertex(v)
        };
        self.set_vertex(v, Some(vertex));
    }

    pub(crate) fn set_next(&mut self, h: HalfEdgeId, next: HalfEdgeId) {
        let a = HalfEdge {
            next,
            ..*self.he(h)
        };
        self.set_halfedge(h, Some(a));
        let b = HalfEdge {
            prev: h,
            ..*self.he(next)
        };
        self.set_halfedge(next, Some(b));
    }

    pub(crate) fn set_outgoing(&mut self, v: VertexId, outgoing: Option<HalfEdgeId>) {
        let vertex = Vertex {
            outgoing,
            ..*self.vertex(v)
        };
        self.set_vertex(v, Some(vertex));
    }

    pub(crate) fn set_face_of(&mut self, h: HalfEdgeId, face: Option<FaceId>) {
        let value = HalfEdge {
            face,
            ..*self.he(h)
        };
        self.set_halfedge(h, Some(value));
    }

    // ---- Allocation ------------------------------------------------------

    pub fn add_vertex(&mut self, position: Vec3) -> VertexId {
        let id = take_free(&mut self.free_vertices, &self.vertices)
            .unwrap_or(self.vertices.len() as u32);
        let v = VertexId(id);
        self.set_vertex(
            v,
            Some(Vertex {
                position,
                outgoing: None,
            }),
        );
        v
    }

    pub(crate) fn alloc_edge_slot(&mut self) -> EdgeId {
        let free = {
            let halfedges = &self.halfedges;
            let mut found = None;
            while let Some(e) = self.free_edges.pop() {
                if halfedges.get(e as usize * 2).is_none_or(Option::is_none) {
                    found = Some(e);
                    break;
                }
            }
            found
        };
        EdgeId(free.unwrap_or(self.halfedges.len().div_ceil(2) as u32))
    }

    pub(crate) fn alloc_face_slot(&mut self) -> FaceId {
        FaceId(take_free(&mut self.free_faces, &self.faces).unwrap_or(self.faces.len() as u32))
    }

    // ---- Journal -----------------------------------------------------------

    /// Starts recording changes. Panics if already recording.
    pub(crate) fn begin_journal(&mut self) {
        assert!(self.journal.is_none(), "nested mesh journal");
        self.journal = Some(Vec::new());
    }

    /// Stops recording and returns the changes since [`Self::begin_journal`].
    pub(crate) fn end_journal(&mut self) -> Vec<Change> {
        self.journal.take().expect("journal was started")
    }

    /// Applies recorded changes backwards (`undo`) or forwards.
    pub(crate) fn apply_changes(&mut self, changes: &[Change], undo: bool) {
        let mut apply = |change: &Change| match *change {
            Change::Vertex(i, old, new) => {
                self.set_vertex(VertexId(i), if undo { old } else { new })
            }
            Change::HalfEdge(i, old, new) => {
                self.set_halfedge(HalfEdgeId(i), if undo { old } else { new })
            }
            Change::Face(i, old, new) => self.set_face(FaceId(i), if undo { old } else { new }),
        };
        if undo {
            changes.iter().rev().for_each(&mut apply);
        } else {
            changes.iter().for_each(&mut apply);
        }
    }
}

fn grow<T>(slots: &mut Vec<Option<T>>, index: usize) -> &mut Option<T> {
    if slots.len() <= index {
        slots.resize_with(index + 1, || None);
    }
    &mut slots[index]
}

fn take_free<T>(free: &mut Vec<u32>, slots: &[Option<T>]) -> Option<u32> {
    while let Some(i) = free.pop() {
        if slots.get(i as usize).is_none_or(Option::is_none) {
            return Some(i);
        }
    }
    None
}
