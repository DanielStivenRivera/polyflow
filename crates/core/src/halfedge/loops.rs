//! Edge loops, edge rings and face loops, for selection.
//!
//! An edge loop runs straight through vertices with four edges, the way
//! quad topology flows; on a border it follows the whole border. An edge
//! ring crosses quads from one edge to the opposite one, and the quads it
//! crosses form a face loop.

use std::collections::HashSet;

use super::{EdgeId, FaceId, HalfEdgeId, HalfEdgeMesh, VertexId};

impl HalfEdgeMesh {
    /// Number of edges at `v`.
    pub fn valence(&self, v: VertexId) -> usize {
        self.outgoing_halfedges(v).count()
    }

    /// The edge loop through `e`, in order. A border edge gives its whole
    /// border; an interior loop continues through interior vertices with
    /// four edges and stops at any other vertex (a pole or a border).
    pub fn edge_loop(&self, e: EdgeId) -> Vec<EdgeId> {
        let (h0, h1) = (e.halfedge(0), e.halfedge(1));
        if !self.contains_halfedge(h0) {
            return Vec::new();
        }
        if let Some(border) = [h0, h1].into_iter().find(|&h| self.is_boundary_halfedge(h)) {
            return self.loop_from(border).map(HalfEdgeId::edge).collect();
        }
        self.walk(e, h0, h1, |h| self.loop_step(h).map(|next| (next, None)))
            .0
    }

    /// The edge ring through `e`, in order: from each quad on its sides to
    /// the opposite edge, until a border or a face that is not a quad.
    pub fn edge_ring(&self, e: EdgeId) -> Vec<EdgeId> {
        self.ring(e).0
    }

    /// The quads crossed by the edge ring through `e` (see
    /// [`Self::edge_ring`]), in order.
    pub fn face_loop(&self, e: EdgeId) -> Vec<FaceId> {
        self.ring(e).1
    }

    fn ring(&self, e: EdgeId) -> (Vec<EdgeId>, Vec<FaceId>) {
        let (h0, h1) = (e.halfedge(0), e.halfedge(1));
        if !self.contains_halfedge(h0) {
            return (Vec::new(), Vec::new());
        }
        self.walk(e, h0, h1, |h| {
            let f = self.face_of(h)?;
            (self.face_halfedges(f).count() == 4).then(|| (self.next(self.next(h)).twin(), Some(f)))
        })
    }

    /// The edge leaving `h`'s target straight ahead of `h`, at an interior
    /// vertex with four edges.
    fn loop_step(&self, h: HalfEdgeId) -> Option<HalfEdgeId> {
        let b = self.target(h);
        if self.is_boundary_vertex(b) {
            return None;
        }
        let out: Vec<HalfEdgeId> = self.outgoing_halfedges(b).collect();
        if out.len() != 4 {
            return None;
        }
        let back = out.iter().position(|&x| x == h.twin())?;
        Some(out[(back + 2) % 4])
    }

    /// Walks from `e` forward through `forward` and backward through
    /// `backward` with `step`, which maps a half-edge to the next one (and
    /// a face it crossed). Returns the edges in order, starting at `e` if
    /// the walk closes, and the faces crossed.
    fn walk(
        &self,
        e: EdgeId,
        forward: HalfEdgeId,
        backward: HalfEdgeId,
        step: impl Fn(HalfEdgeId) -> Option<(HalfEdgeId, Option<FaceId>)>,
    ) -> (Vec<EdgeId>, Vec<FaceId>) {
        let mut seen = HashSet::from([e]);
        let mut walk_from = |start: HalfEdgeId| {
            let (mut edges, mut faces) = (Vec::new(), Vec::new());
            let mut h = start;
            let mut closed = false;
            while let Some((next, face)) = step(h) {
                faces.extend(face);
                if next.edge() == e {
                    closed = true;
                    break;
                }
                if !seen.insert(next.edge()) {
                    break;
                }
                edges.push(next.edge());
                h = next;
            }
            (edges, faces, closed)
        };
        let (ahead, ahead_faces, closed) = walk_from(forward);
        let mut edges = vec![e];
        let mut faces = Vec::new();
        if !closed {
            let (behind, behind_faces, _) = walk_from(backward);
            edges = behind.into_iter().rev().chain(edges).collect();
            faces = behind_faces.into_iter().rev().collect();
        }
        edges.extend(ahead);
        faces.extend(ahead_faces);
        (edges, faces)
    }
}
