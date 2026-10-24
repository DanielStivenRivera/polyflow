//! Topological operations. They follow OpenMesh's `PolyConnectivity`, adapted
//! to half-edges that store their origin vertex.

use std::collections::HashSet;

use super::{EdgeId, Face, FaceId, HalfEdge, HalfEdgeId, HalfEdgeMesh, VertexId};

/// Why a topological operation was refused. The mesh is left unchanged.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TopologyError {
    #[error("a face needs at least 3 vertices")]
    TooFewVertices,
    #[error("the face uses a vertex more than once")]
    RepeatedVertex,
    #[error("vertex {0:?} does not exist")]
    MissingVertex(VertexId),
    #[error("face {0:?} does not exist")]
    MissingFace(FaceId),
    #[error("vertex {0:?} is surrounded by faces; a new face would make it non-manifold")]
    InteriorVertex(VertexId),
    #[error("the edge {0:?} -> {1:?} already has a face on that side (is the face flipped?)")]
    EdgeTaken(VertexId, VertexId),
    #[error("the face would close a fan around a vertex that other faces also touch")]
    PatchRelinkFailed,
}

impl HalfEdgeMesh {
    /// Adds a polygon through `vertices`, in counter-clockwise order seen from
    /// its front. It may share edges with existing faces (with opposite
    /// orientation) and attach to boundary vertices; anything that would make
    /// the mesh non-manifold is refused and leaves the mesh unchanged.
    pub fn add_face(&mut self, vertices: &[VertexId]) -> Result<FaceId, TopologyError> {
        let n = vertices.len();
        if n < 3 {
            return Err(TopologyError::TooFewVertices);
        }
        for &v in vertices {
            if !self.contains_vertex(v) {
                return Err(TopologyError::MissingVertex(v));
            }
        }
        if vertices.iter().collect::<HashSet<_>>().len() != n {
            return Err(TopologyError::RepeatedVertex);
        }

        // Check for topological errors before touching anything.
        let mut halfedges: Vec<Option<HalfEdgeId>> = Vec::with_capacity(n);
        for i in 0..n {
            let (v, w) = (vertices[i], vertices[(i + 1) % n]);
            if !self.is_boundary_vertex(v) {
                return Err(TopologyError::InteriorVertex(v));
            }
            let h = self.find_halfedge(v, w);
            if h.is_some_and(|h| !self.is_boundary_halfedge(h)) {
                return Err(TopologyError::EdgeTaken(v, w));
            }
            halfedges.push(h);
        }

        // Where two consecutive edges already exist but are not consecutive in
        // their hole, the patch between them must be moved elsewhere around
        // the vertex. Compute the relinking first so failures change nothing.
        let mut next_cache: Vec<(HalfEdgeId, HalfEdgeId)> = Vec::with_capacity(3 * n);
        for i in 0..n {
            let ii = (i + 1) % n;
            let (Some(inner_prev), Some(inner_next)) = (halfedges[i], halfedges[ii]) else {
                continue;
            };
            if self.next(inner_prev) == inner_next {
                continue;
            }
            // The vertex has several fans, and the fans between `inner_prev`'s
            // and `inner_next`'s (the patch) must move to another gap. Rotate
            // through `inner_next`'s fan to its incoming boundary half-edge:
            // the patch goes into the gap after it.
            let outer_prev = inner_next.twin();
            let mut boundary_prev = outer_prev;
            let mut steps = 0;
            loop {
                boundary_prev = self.next(boundary_prev).twin();
                steps += 1;
                if self.is_boundary_halfedge(boundary_prev) {
                    break;
                }
                if boundary_prev == outer_prev || steps > self.halfedges.len() {
                    return Err(TopologyError::PatchRelinkFailed);
                }
            }
            // Both edges belong to the same fan: the face would close it into
            // a disk, leaving the vertex's other fans nowhere to attach.
            if boundary_prev == inner_prev {
                return Err(TopologyError::PatchRelinkFailed);
            }
            let boundary_next = self.next(boundary_prev);
            if boundary_next == inner_next {
                return Err(TopologyError::PatchRelinkFailed);
            }
            let patch_start = self.next(inner_prev);
            let patch_end = self.prev(inner_next);
            next_cache.push((boundary_prev, patch_start));
            next_cache.push((patch_end, boundary_next));
            next_cache.push((inner_prev, inner_next));
        }

        // From here on the operation cannot fail.
        let is_new: Vec<bool> = halfedges.iter().map(Option::is_none).collect();
        let halfedges: Vec<HalfEdgeId> = (0..n)
            .map(|i| {
                halfedges[i].unwrap_or_else(|| self.new_edge(vertices[i], vertices[(i + 1) % n]))
            })
            .collect();

        let f = self.alloc_face_slot();
        self.set_face(
            f,
            Some(Face {
                halfedge: halfedges[0],
            }),
        );

        let mut needs_adjust = vec![false; n];
        for i in 0..n {
            let ii = (i + 1) % n;
            // The vertex between `inner_prev` and `inner_next`.
            let v = vertices[ii];
            let inner_prev = halfedges[i];
            let inner_next = halfedges[ii];
            match (is_new[i], is_new[ii]) {
                (false, false) => needs_adjust[ii] = self.outgoing(v) == Some(inner_next),
                (prev_new, next_new) => {
                    let outer_prev = inner_next.twin();
                    let outer_next = inner_prev.twin();
                    match (prev_new, next_new) {
                        (true, false) => {
                            let boundary_prev = self.prev(inner_next);
                            next_cache.push((boundary_prev, outer_next));
                            self.set_outgoing(v, Some(outer_next));
                        }
                        (false, true) => {
                            let boundary_next = self.next(inner_prev);
                            next_cache.push((outer_prev, boundary_next));
                            self.set_outgoing(v, Some(boundary_next));
                        }
                        _ => match self.outgoing(v) {
                            None => {
                                self.set_outgoing(v, Some(outer_next));
                                next_cache.push((outer_prev, outer_next));
                            }
                            Some(boundary_next) => {
                                let boundary_prev = self.prev(boundary_next);
                                next_cache.push((boundary_prev, outer_next));
                                next_cache.push((outer_prev, boundary_next));
                            }
                        },
                    }
                    next_cache.push((inner_prev, inner_next));
                }
            }
            self.set_face_of(inner_prev, Some(f));
        }

        for (h, next) in next_cache {
            self.set_next(h, next);
        }
        for (i, &adjust) in needs_adjust.iter().enumerate() {
            if adjust {
                self.adjust_outgoing(vertices[i]);
            }
        }
        Ok(f)
    }

    /// Removes a face. Edges left without any face are removed too; vertices
    /// left without edges are removed if `remove_isolated_vertices`.
    pub fn remove_face(
        &mut self,
        f: FaceId,
        remove_isolated_vertices: bool,
    ) -> Result<(), TopologyError> {
        if !self.contains_face(f) {
            return Err(TopologyError::MissingFace(f));
        }
        let halfedges: Vec<HalfEdgeId> = self.face_halfedges(f).collect();
        let vertices: Vec<VertexId> = halfedges.iter().map(|&h| self.origin(h)).collect();

        let mut dead_edges = Vec::new();
        for &h in &halfedges {
            self.set_face_of(h, None);
            if self.is_boundary_halfedge(h.twin()) {
                dead_edges.push(h.edge());
            }
        }
        self.set_face(f, None);

        for e in dead_edges {
            let (h0, h1) = (e.halfedge(0), e.halfedge(1));
            let (a, b) = (self.origin(h0), self.origin(h1));
            let (next0, prev0) = (self.next(h0), self.prev(h0));
            let (next1, prev1) = (self.next(h1), self.prev(h1));
            self.set_next(prev0, next1);
            self.set_next(prev1, next0);
            // `next0` leaves `b` and `next1` leaves `a`, unless the edge was alone.
            if self.outgoing(b) == Some(h1) {
                self.set_outgoing(b, (next0 != h1).then_some(next0));
            }
            if self.outgoing(a) == Some(h0) {
                self.set_outgoing(a, (next1 != h0).then_some(next1));
            }
            self.delete_edge(e);
        }

        for v in vertices {
            if self.outgoing(v).is_none() {
                if remove_isolated_vertices {
                    self.set_vertex(v, None);
                }
            } else {
                self.adjust_outgoing(v);
            }
        }
        Ok(())
    }

    /// Removes a vertex and every face around it.
    pub fn remove_vertex(&mut self, v: VertexId) -> Result<(), TopologyError> {
        if !self.contains_vertex(v) {
            return Err(TopologyError::MissingVertex(v));
        }
        let faces: Vec<FaceId> = self.vertex_faces(v).collect();
        for f in faces {
            self.remove_face(f, false)?;
        }
        debug_assert!(self.outgoing(v).is_none(), "no loose edges remain");
        self.set_vertex(v, None);
        Ok(())
    }

    /// Creates the half-edge pair `from -> to` / `to -> from`, linked to each
    /// other; returns the `from -> to` half.
    fn new_edge(&mut self, from: VertexId, to: VertexId) -> HalfEdgeId {
        let e = self.alloc_edge_slot();
        let (h0, h1) = (e.halfedge(0), e.halfedge(1));
        // Odd half first: edge bookkeeping happens when the even half appears.
        self.set_halfedge(
            h1,
            Some(HalfEdge {
                origin: to,
                next: h0,
                prev: h0,
                face: None,
                uv: None,
            }),
        );
        self.set_halfedge(
            h0,
            Some(HalfEdge {
                origin: from,
                next: h1,
                prev: h1,
                face: None,
                uv: None,
            }),
        );
        h0
    }

    fn delete_edge(&mut self, e: EdgeId) {
        self.set_halfedge(e.halfedge(0), None);
        self.set_halfedge(e.halfedge(1), None);
    }

    /// Points `v.outgoing` at a boundary half-edge if there is one.
    fn adjust_outgoing(&mut self, v: VertexId) {
        let boundary = self
            .outgoing_halfedges(v)
            .find(|&h| self.is_boundary_halfedge(h));
        if let Some(h) = boundary {
            self.set_outgoing(v, Some(h));
        }
    }
}
