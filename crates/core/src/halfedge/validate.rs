//! Structural invariant checks, for tests and debug assertions.

use std::collections::HashSet;

use super::{FaceId, HalfEdgeId, HalfEdgeMesh, VertexId};

impl HalfEdgeMesh {
    /// Checks every invariant listed in the [module docs](super) and that the
    /// cached counts are right. Returns a description of the first problem.
    pub fn validate(&self) -> Result<(), String> {
        let check = |ok: bool, what: String| if ok { Ok(()) } else { Err(what) };

        // Half-edges: pairing, links, faces.
        for h in self.halfedges() {
            let he = self.he(h);
            check(
                self.contains_halfedge(h.twin()),
                format!("{h:?}: twin is missing"),
            )?;
            check(
                self.contains_halfedge(he.next),
                format!("{h:?}: next is missing"),
            )?;
            check(
                self.contains_halfedge(he.prev),
                format!("{h:?}: prev is missing"),
            )?;
            check(
                self.contains_vertex(he.origin),
                format!("{h:?}: origin is missing"),
            )?;
            check(
                self.prev(he.next) == h,
                format!("{h:?}: prev(next(h)) != h"),
            )?;
            check(
                self.next(he.prev) == h,
                format!("{h:?}: next(prev(h)) != h"),
            )?;
            check(
                self.origin(he.next) == self.target(h),
                format!("{h:?}: next does not start at h's target"),
            )?;
            check(
                self.face_of(he.next) == he.face,
                format!("{h:?}: next is in a different face"),
            )?;
            check(he.origin != self.target(h), format!("{h:?}: loop edge"))?;
            if let Some(f) = he.face {
                check(
                    self.contains_face(f),
                    format!("{h:?}: face {f:?} is missing"),
                )?;
            }
            check(
                he.face.is_some() || he.uv.is_none(),
                format!("{h:?}: boundary half-edge has a UV"),
            )?;
            check(
                he.face.is_some() || self.face_of(h.twin()).is_some(),
                format!("{h:?}: loose edge (no face on either side)"),
            )?;
        }

        // Faces: closed loops of at least 3 half-edges pointing back at the face.
        for f in self.faces() {
            let start = self.face_halfedge(f);
            check(
                self.contains_halfedge(start),
                format!("{f:?}: half-edge is missing"),
            )?;
            check(
                self.face_of(start) == Some(f),
                format!("{f:?}: its half-edge belongs elsewhere"),
            )?;
            let len = self
                .bounded_loop_len(start)
                .ok_or(format!("{f:?}: loop does not close"))?;
            check(len >= 3, format!("{f:?}: only {len} sides"))?;
        }

        // Vertices: outgoing half-edge, single fan, boundary preference.
        let mut outgoing_by_vertex: Vec<Vec<HalfEdgeId>> = vec![Vec::new(); self.vertices.len()];
        for h in self.halfedges() {
            outgoing_by_vertex[self.origin(h).index()].push(h);
        }
        for v in self.vertices() {
            let all = &outgoing_by_vertex[v.index()];
            let Some(start) = self.outgoing(v) else {
                check(
                    all.is_empty(),
                    format!("{v:?}: has edges but no outgoing half-edge"),
                )?;
                continue;
            };
            check(
                self.contains_halfedge(start),
                format!("{v:?}: outgoing half-edge is missing"),
            )?;
            check(
                self.origin(start) == v,
                format!("{v:?}: outgoing half-edge starts elsewhere"),
            )?;
            let fan: HashSet<HalfEdgeId> = self
                .bounded_fan(v)
                .ok_or(format!("{v:?}: fan does not close"))?;
            check(
                fan.len() == all.len(),
                format!(
                    "{v:?}: rotation reaches {} of its {} edges",
                    fan.len(),
                    all.len()
                ),
            )?;
            let boundary = all.iter().any(|&h| self.is_boundary_halfedge(h));
            check(
                !boundary || self.is_boundary_halfedge(start),
                format!("{v:?}: on a boundary but outgoing is not a boundary half-edge"),
            )?;
        }

        // Cached counts.
        check(
            self.vertices().count() == self.vertex_count,
            "vertex count is stale".into(),
        )?;
        check(
            self.faces().count() == self.face_count,
            "face count is stale".into(),
        )?;
        check(
            self.edges().count() == self.edge_count,
            "edge count is stale".into(),
        )?;
        let unique: HashSet<FaceId> = self.faces().collect();
        check(unique.len() == self.face_count, "duplicate faces".into())?;
        Ok(())
    }

    /// Length of the `next` loop from `start`, or `None` if it does not close
    /// within the number of half-edges.
    fn bounded_loop_len(&self, start: HalfEdgeId) -> Option<usize> {
        let mut h = start;
        for len in 1..=self.halfedges.len() {
            h = self.next(h);
            if h == start {
                return Some(len);
            }
        }
        None
    }

    fn bounded_fan(&self, v: VertexId) -> Option<HashSet<HalfEdgeId>> {
        let start = self.outgoing(v)?;
        let mut fan = HashSet::new();
        let mut h = start;
        for _ in 0..=self.halfedges.len() {
            if !fan.insert(h) {
                return None; // Revisited a half-edge without returning to `start`.
            }
            h = self.next(h.twin());
            if h == start {
                return Some(fan);
            }
        }
        None
    }
}
