//! Topological edits for retopology: splitting, merging, dissolving and
//! rotating, and inserting edge loops.
//!
//! Each operation rebuilds the faces it touches: it removes them and adds
//! their replacements with [`HalfEdgeMesh::add_face`], so it gets the same
//! manifold checks. Face corners keep their UVs; corners of new vertices get
//! UVs interpolated along the edge they split. An operation that fails may
//! leave the mesh half-edited: run it inside `MeshDocument::edit`, which
//! rolls it back.

use std::collections::{HashMap, HashSet};

use glam::Vec2;

use super::{EdgeId, FaceId, HalfEdgeId, HalfEdgeMesh, TopologyError, VertexId};

/// A face corner: its vertex and its UV in that face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    pub vertex: VertexId,
    pub uv: Option<Vec2>,
}

impl HalfEdgeMesh {
    /// The corners of a face, in [`Self::face_vertices`] order.
    pub fn face_corners(&self, f: FaceId) -> Vec<Corner> {
        self.face_halfedges(f)
            .map(|h| Corner {
                vertex: self.origin(h),
                uv: self.uv(h),
            })
            .collect()
    }

    /// Splits the edge of `h` with a new vertex at `t` along it (0 at `h`'s
    /// origin). Faces on both sides get the vertex.
    pub fn split_edge(&mut self, h: HalfEdgeId, t: f32) -> Result<VertexId, TopologyError> {
        self.check_halfedge(h)?;
        let (a, b) = (self.origin(h), self.target(h));
        let m = self.add_vertex(self.position(a).lerp(self.position(b), t));
        let faces: Vec<FaceId> = [h, h.twin()]
            .into_iter()
            .filter_map(|h| self.face_of(h))
            .collect();
        let new: Vec<Vec<Corner>> = faces
            .iter()
            .map(|&f| insert_on_edge(&self.face_corners(f), a, b, m, t))
            .collect();
        self.rebuild(&faces, new)?;
        Ok(m)
    }

    /// Splits a face in two along a new edge between two of its vertices,
    /// which must not be neighbors in it. Returns the new edge.
    pub fn split_face(
        &mut self,
        f: FaceId,
        a: VertexId,
        b: VertexId,
    ) -> Result<EdgeId, TopologyError> {
        if !self.contains_face(f) {
            return Err(TopologyError::MissingFace(f));
        }
        let corners = self.face_corners(f);
        let n = corners.len();
        let index = |v| corners.iter().position(|c| c.vertex == v);
        let (Some(ia), Some(ib)) = (index(a), index(b)) else {
            return Err(TopologyError::Refused("both vertices must be on the face"));
        };
        if ia == ib || (ia + 1) % n == ib || (ib + 1) % n == ia {
            return Err(TopologyError::Refused(
                "the vertices are already joined by an edge",
            ));
        }
        let new = vec![cyclic(&corners, ia, ib), cyclic(&corners, ib, ia)];
        self.rebuild(&[f], new)?;
        Ok(self.find_halfedge(a, b).expect("just added").edge())
    }

    /// Merges `vertices` into the first of them, moved to `position`. Faces
    /// collapsed to fewer than 3 vertices disappear. Returns the vertex kept.
    pub fn merge_vertices(
        &mut self,
        vertices: &[VertexId],
        position: glam::Vec3,
    ) -> Result<VertexId, TopologyError> {
        let Some(&keep) = vertices.first() else {
            return Err(TopologyError::Refused("nothing to merge"));
        };
        for &v in vertices {
            if !self.contains_vertex(v) {
                return Err(TopologyError::MissingVertex(v));
            }
        }
        let merged: HashSet<VertexId> = vertices.iter().copied().collect();
        let mut faces: Vec<FaceId> = merged.iter().flat_map(|&v| self.vertex_faces(v)).collect();
        faces.sort_unstable();
        faces.dedup();
        let mut new = Vec::new();
        let mut seen: HashSet<Vec<VertexId>> = HashSet::new();
        for &f in &faces {
            let mut corners = self.face_corners(f);
            for c in &mut corners {
                if merged.contains(&c.vertex) {
                    c.vertex = keep;
                }
            }
            // Drop corners that collapsed onto the previous one.
            let mut kept: Vec<Corner> = Vec::with_capacity(corners.len());
            for (i, &c) in corners.iter().enumerate() {
                let prev = corners[(i + corners.len() - 1) % corners.len()];
                if c.vertex != prev.vertex {
                    kept.push(c);
                }
            }
            if kept.len() < 3 {
                continue;
            }
            if kept.iter().filter(|c| c.vertex == keep).count() > 1 {
                return Err(TopologyError::Refused(
                    "merging would make a face use a vertex twice",
                ));
            }
            // Two faces collapsed onto the same vertices: keep one.
            let mut key: Vec<VertexId> = kept.iter().map(|c| c.vertex).collect();
            key.sort_unstable();
            if seen.insert(key) {
                new.push(kept);
            }
        }
        self.rebuild(&faces, new)?;
        if self.contains_vertex(keep) {
            self.set_position(keep, position);
        }
        for &v in &merged {
            if v != keep && self.contains_vertex(v) && self.outgoing(v).is_none() {
                self.set_vertex(v, None);
            }
        }
        Ok(keep)
    }

    /// Merges each connected group of `faces` (connected through shared
    /// edges) into one polygon. A group must be a disk: one with a hole has
    /// no single outline.
    pub fn dissolve_faces(&mut self, faces: &[FaceId]) -> Result<Vec<FaceId>, TopologyError> {
        let set: HashSet<FaceId> = faces.iter().copied().collect();
        for &f in &set {
            if !self.contains_face(f) {
                return Err(TopologyError::MissingFace(f));
            }
        }
        // Connected groups by flood fill across shared edges.
        let mut groups: Vec<Vec<FaceId>> = Vec::new();
        let mut assigned: HashSet<FaceId> = HashSet::new();
        let mut sorted: Vec<FaceId> = set.iter().copied().collect();
        sorted.sort_unstable();
        for f in sorted {
            if !assigned.insert(f) {
                continue;
            }
            let mut group = vec![f];
            let mut i = 0;
            while i < group.len() {
                let g = group[i];
                for h in self.face_halfedges(g) {
                    if let Some(n) = self.face_of(h.twin())
                        && set.contains(&n)
                        && assigned.insert(n)
                    {
                        group.push(n);
                    }
                }
                i += 1;
            }
            groups.push(group);
        }
        self.merge_groups(groups)
    }

    /// Merges each group of faces into one polygon.
    fn merge_groups(&mut self, groups: Vec<Vec<FaceId>>) -> Result<Vec<FaceId>, TopologyError> {
        let mut removed = Vec::new();
        let mut new = Vec::new();
        for group in groups {
            if group.len() == 1 {
                continue;
            }
            new.push(self.outline(&group)?);
            removed.extend(group);
        }
        self.rebuild(&removed, new)
    }

    /// Removes edges, merging the faces on both sides. Vertices left inside
    /// a merged face are removed, and so are vertices left between just two
    /// edges in the middle of a face border. Border edges are skipped.
    pub fn dissolve_edges(&mut self, edges: &[EdgeId]) -> Result<(), TopologyError> {
        let mut parent: HashMap<FaceId, FaceId> = HashMap::new();
        fn root(parent: &mut HashMap<FaceId, FaceId>, f: FaceId) -> FaceId {
            let p = *parent.entry(f).or_insert(f);
            if p == f {
                return f;
            }
            let r = root(parent, p);
            parent.insert(f, r);
            r
        }
        let mut ends = Vec::new();
        for &e in edges {
            let (h0, h1) = (e.halfedge(0), e.halfedge(1));
            self.check_halfedge(h0)?;
            let (Some(f), Some(g)) = (self.face_of(h0), self.face_of(h1)) else {
                continue;
            };
            let (rf, rg) = (root(&mut parent, f), root(&mut parent, g));
            parent.insert(rf, rg);
            ends.extend([self.origin(h0), self.origin(h1)]);
        }
        let mut groups: HashMap<FaceId, Vec<FaceId>> = HashMap::new();
        let mut faces: Vec<FaceId> = parent.keys().copied().collect();
        faces.sort_unstable();
        for f in faces {
            let r = root(&mut parent, f);
            groups.entry(r).or_default().push(f);
        }
        let mut groups: Vec<Vec<FaceId>> = groups.into_values().collect();
        groups.sort_unstable();
        self.merge_groups(groups)?;
        for v in ends {
            if self.contains_vertex(v) && !self.is_boundary_vertex(v) && self.valence(v) == 2 {
                self.dissolve_vertex(v)?;
            }
        }
        Ok(())
    }

    /// Removes a vertex, merging the faces around it into one. A vertex
    /// between just two edges is taken out of its faces instead.
    pub fn dissolve_vertex(&mut self, v: VertexId) -> Result<(), TopologyError> {
        if !self.contains_vertex(v) {
            return Err(TopologyError::MissingVertex(v));
        }
        let faces: Vec<FaceId> = self.vertex_faces(v).collect();
        let new: Vec<Vec<Corner>> = if self.valence(v) == 2 || faces.len() == 1 {
            faces
                .iter()
                .map(|&f| {
                    let mut corners = self.face_corners(f);
                    corners.retain(|c| c.vertex != v);
                    corners
                })
                .collect()
        } else {
            let mut outline = self.outline(&faces)?;
            outline.retain(|c| c.vertex != v);
            vec![outline]
        };
        if new.iter().any(|c| c.len() < 3) {
            return Err(TopologyError::Refused(
                "dissolving the vertex would leave a face with fewer than 3 sides",
            ));
        }
        self.rebuild(&faces, new)?;
        if self.contains_vertex(v) && self.outgoing(v).is_none() {
            self.set_vertex(v, None);
        }
        Ok(())
    }

    /// Turns an edge between two faces one step around the polygon they
    /// form (counter-clockwise, or clockwise if not `ccw`), so it joins the
    /// next pair of vertices. Returns the new edge.
    pub fn rotate_edge(&mut self, e: EdgeId, ccw: bool) -> Result<EdgeId, TopologyError> {
        let (h0, h1) = (e.halfedge(0), e.halfedge(1));
        self.check_halfedge(h0)?;
        let (Some(f), Some(g)) = (self.face_of(h0), self.face_of(h1)) else {
            return Err(TopologyError::Refused("a border edge cannot be rotated"));
        };
        let (a, b) = (self.origin(h0), self.origin(h1));
        if f == g {
            return Err(TopologyError::Refused(
                "the edge has the same face on both sides",
            ));
        }
        let outline = self.outline(&[f, g])?;
        let n = outline.len();
        let index = |v| outline.iter().position(|c: &Corner| c.vertex == v);
        // Faces sharing more than this edge leave an end inside their outline.
        let (Some(ia), Some(ib)) = (index(a), index(b)) else {
            return Err(TopologyError::Refused("the faces share more than one edge"));
        };
        let step = if ccw { 1 } else { n - 1 };
        let (ia, ib) = ((ia + step) % n, (ib + step) % n);
        let (x, y) = (outline[ia].vertex, outline[ib].vertex);
        if self.find_halfedge(x, y).is_some() {
            return Err(TopologyError::Refused(
                "the rotated edge would duplicate an existing edge",
            ));
        }
        let new = vec![cyclic(&outline, ia, ib), cyclic(&outline, ib, ia)];
        if new.iter().any(|c| c.len() < 3) {
            return Err(TopologyError::Refused("the faces are too small to rotate"));
        }
        self.rebuild(&[f, g], new)?;
        Ok(self.find_halfedge(x, y).expect("just added").edge())
    }

    /// The half-edges of the edge ring through `h`'s edge (see
    /// [`Self::edge_ring`]), all pointing the same way as `h`, in order, and
    /// whether the ring is closed. Consecutive half-edges are opposite sides
    /// of a quad, the face of the first one.
    pub fn ring_halfedges(&self, h: HalfEdgeId) -> (Vec<HalfEdgeId>, bool) {
        let step = |h: HalfEdgeId| -> Option<HalfEdgeId> {
            let f = self.face_of(h)?;
            (self.face_halfedges(f).count() == 4).then(|| self.next(self.next(h)).twin())
        };
        let mut seen = HashSet::from([h.edge()]);
        let mut ahead = Vec::new();
        let mut current = h;
        while let Some(next) = step(current) {
            if next.edge() == h.edge() {
                return (std::iter::once(h).chain(ahead).collect(), true);
            }
            if !seen.insert(next.edge()) {
                break;
            }
            ahead.push(next);
            current = next;
        }
        let mut behind = Vec::new();
        let mut current = h.twin();
        while let Some(next) = step(current) {
            if !seen.insert(next.edge()) {
                break;
            }
            behind.push(next.twin());
            current = next;
        }
        let ring = behind.into_iter().rev().chain([h]).chain(ahead).collect();
        (ring, false)
    }

    /// Inserts an edge loop across the edge ring through `h`'s edge: each
    /// ring edge is split at `t` (measured in `h`'s direction) and each quad
    /// of the ring is split in two. Faces at the ends of an open ring get
    /// the new vertex on their side. Returns the new vertices in ring order.
    pub fn insert_loop(&mut self, h: HalfEdgeId, t: f32) -> Result<Vec<VertexId>, TopologyError> {
        self.check_halfedge(h)?;
        let (ring, closed) = self.ring_halfedges(h);
        let ends: Vec<(VertexId, VertexId)> = ring
            .iter()
            .map(|&h| (self.origin(h), self.target(h)))
            .collect();
        let mids: Vec<VertexId> = ends
            .iter()
            .map(|&(a, b)| self.add_vertex(self.position(a).lerp(self.position(b), t)))
            .collect();
        let pairs = if closed { ring.len() } else { ring.len() - 1 };
        let mut crossed: Vec<FaceId> = Vec::with_capacity(pairs);
        let mut new: Vec<Vec<Corner>> = Vec::new();
        for i in 0..pairs {
            let j = (i + 1) % ring.len();
            let f = self.face_of(ring[i]).expect("ring quads have faces");
            if crossed.contains(&f) {
                return Err(TopologyError::Refused("the ring crosses a face twice"));
            }
            crossed.push(f);
            // f = [a_i, b_i, b_j, a_j]; split between the new vertices.
            let corners = self.face_corners(f);
            let at = |v| {
                corners[corners
                    .iter()
                    .position(|c| c.vertex == v)
                    .expect("quad corner")]
            };
            let (ai, bi) = (at(ends[i].0), at(ends[i].1));
            let (aj, bj) = (at(ends[j].0), at(ends[j].1));
            let mi = Corner {
                vertex: mids[i],
                uv: lerp_uv(ai.uv, bi.uv, t),
            };
            let mj = Corner {
                vertex: mids[j],
                uv: lerp_uv(aj.uv, bj.uv, t),
            };
            new.push(vec![ai, mi, mj, aj]);
            new.push(vec![mi, bi, bj, mj]);
        }
        // Other faces along the ring edges (at the ends of an open ring).
        let mut others: Vec<FaceId> = Vec::new();
        for &h in &ring {
            for side in [h, h.twin()] {
                if let Some(f) = self.face_of(side)
                    && !crossed.contains(&f)
                    && !others.contains(&f)
                {
                    others.push(f);
                }
            }
        }
        for &f in &others {
            let mut corners = self.face_corners(f);
            for (k, &(a, b)) in ends.iter().enumerate() {
                if corners.iter().any(|c| c.vertex == a) && corners.iter().any(|c| c.vertex == b) {
                    corners = insert_on_edge(&corners, a, b, mids[k], t);
                }
            }
            new.push(corners);
        }
        let removed: Vec<FaceId> = crossed.into_iter().chain(others).collect();
        self.rebuild(&removed, new)?;
        Ok(mids)
    }

    /// The outline of a connected group of faces, as corners of the faces
    /// along it, counter-clockwise.
    fn outline(&self, faces: &[FaceId]) -> Result<Vec<Corner>, TopologyError> {
        let set: HashSet<FaceId> = faces.iter().copied().collect();
        let mut by_origin: HashMap<VertexId, HalfEdgeId> = HashMap::new();
        for &f in faces {
            for h in self.face_halfedges(f) {
                if self.face_of(h.twin()).is_some_and(|g| set.contains(&g)) {
                    continue;
                }
                if by_origin.insert(self.origin(h), h).is_some() {
                    return Err(TopologyError::Refused(
                        "the faces touch at a corner; their outline is not a simple polygon",
                    ));
                }
            }
        }
        let Some(&start) = by_origin.values().min() else {
            return Err(TopologyError::Refused("the faces enclose nothing"));
        };
        let mut outline = Vec::new();
        let mut h = start;
        loop {
            outline.push(Corner {
                vertex: self.origin(h),
                uv: self.uv(h),
            });
            h = by_origin[&self.target(h)];
            if h == start {
                break;
            }
            if outline.len() > by_origin.len() {
                unreachable!("outline loops are bounded by their half-edges");
            }
        }
        if outline.len() != by_origin.len() {
            return Err(TopologyError::Refused(
                "the faces surround a hole; their outline is not a single polygon",
            ));
        }
        Ok(outline)
    }

    /// Replaces faces: removes `remove`, then adds `add` with their corner
    /// UVs. Vertices of removed faces that end up without edges are removed.
    fn rebuild(
        &mut self,
        remove: &[FaceId],
        add: Vec<Vec<Corner>>,
    ) -> Result<Vec<FaceId>, TopologyError> {
        let mut touched: Vec<VertexId> = Vec::new();
        for &f in remove {
            touched.extend(self.face_vertices(f));
        }
        for &f in remove {
            self.remove_face(f, false)?;
        }
        // Some faces can only be added once their neighbors are there (a
        // vertex may not get a second fan that closes the first); retry
        // until no more can be added.
        let mut pending: Vec<Vec<Corner>> = add;
        let mut added = Vec::new();
        loop {
            let mut failed = Vec::new();
            let mut error = None;
            let before = pending.len();
            for corners in pending {
                let vertices: Vec<VertexId> = corners.iter().map(|c| c.vertex).collect();
                match self.add_face(&vertices) {
                    Ok(f) => {
                        for (h, c) in self
                            .face_halfedges(f)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .zip(&corners)
                        {
                            if c.uv.is_some() {
                                self.set_uv(h, c.uv);
                            }
                        }
                        added.push(f);
                    }
                    Err(e) => {
                        error = Some(e);
                        failed.push(corners);
                    }
                }
            }
            if failed.is_empty() {
                break;
            }
            if failed.len() == before {
                return Err(error.expect("a face failed"));
            }
            pending = failed;
        }
        for v in touched {
            if self.contains_vertex(v) && self.outgoing(v).is_none() {
                self.set_vertex(v, None);
            }
        }
        Ok(added)
    }

    fn check_halfedge(&self, h: HalfEdgeId) -> Result<(), TopologyError> {
        if self.contains_halfedge(h) {
            Ok(())
        } else {
            Err(TopologyError::Refused("the edge does not exist"))
        }
    }
}

/// Corners from index `from` to index `to`, both included, wrapping around.
fn cyclic(corners: &[Corner], from: usize, to: usize) -> Vec<Corner> {
    let n = corners.len();
    let len = (to + n - from) % n + 1;
    (0..len).map(|k| corners[(from + k) % n]).collect()
}

/// `corners` with `m` inserted on the side between `a` and `b` (in either
/// order); `t` runs from `a` to `b`.
fn insert_on_edge(
    corners: &[Corner],
    a: VertexId,
    b: VertexId,
    m: VertexId,
    t: f32,
) -> Vec<Corner> {
    let n = corners.len();
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let (c, next) = (corners[i], corners[(i + 1) % n]);
        out.push(c);
        let uv = if (c.vertex, next.vertex) == (a, b) {
            Some(lerp_uv(c.uv, next.uv, t))
        } else if (c.vertex, next.vertex) == (b, a) {
            Some(lerp_uv(c.uv, next.uv, 1.0 - t))
        } else {
            None
        };
        if let Some(uv) = uv {
            out.push(Corner { vertex: m, uv });
        }
    }
    out
}

fn lerp_uv(a: Option<Vec2>, b: Option<Vec2>, t: f32) -> Option<Vec2> {
    Some(a?.lerp(b?, t))
}
