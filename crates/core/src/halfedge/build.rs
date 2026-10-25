//! Building new faces from borders: extruding them, bridging two of them,
//! and filling holes.
//!
//! These work on boundary half-edges (the half-edges of an edge that have no
//! face). New faces use them in their direction, so they are oriented like
//! the faces next to them. Like the operations in `topology`, they can leave
//! the mesh half-edited on failure: run them inside `MeshDocument::edit`.

use std::collections::{HashMap, HashSet};
use std::f32::consts::FRAC_PI_2;

use glam::Vec3;

use super::topology::Corner;
use super::{EdgeId, HalfEdgeId, HalfEdgeMesh, TopologyError, VertexId};

impl HalfEdgeMesh {
    /// The boundary half-edges of `edges` (border edges; others are
    /// ignored), linked into chains along their border, each in order. A
    /// chain going all around its hole starts anywhere and is closed.
    pub fn border_chains(&self, edges: &[EdgeId]) -> Vec<Vec<HalfEdgeId>> {
        let set: HashSet<HalfEdgeId> = edges
            .iter()
            .flat_map(|e| [e.halfedge(0), e.halfedge(1)])
            .filter(|&h| self.contains_halfedge(h) && self.is_boundary_halfedge(h))
            .collect();
        let mut sorted: Vec<HalfEdgeId> = set.iter().copied().collect();
        sorted.sort_unstable();
        let mut used: HashSet<HalfEdgeId> = HashSet::new();
        let mut chains = Vec::new();
        // Open chains first, from the half-edges with no predecessor.
        for &h in &sorted {
            if used.contains(&h) || set.contains(&self.prev(h)) {
                continue;
            }
            let mut chain = vec![h];
            used.insert(h);
            let mut current = h;
            while set.contains(&self.next(current)) && used.insert(self.next(current)) {
                current = self.next(current);
                chain.push(current);
            }
            chains.push(chain);
        }
        // What is left goes all around its hole.
        for &h in &sorted {
            if used.contains(&h) {
                continue;
            }
            let chain: Vec<HalfEdgeId> = self.loop_from(h).collect();
            used.extend(chain.iter().copied());
            chains.push(chain);
        }
        chains
    }

    /// Whether a chain of boundary half-edges goes all around its hole.
    pub fn is_closed_chain(&self, chain: &[HalfEdgeId]) -> bool {
        match (chain.first(), chain.last()) {
            (Some(&first), Some(&last)) => self.target(last) == self.origin(first),
            _ => false,
        }
    }

    /// The vertices of a chain, in order (for an open chain, one more than
    /// its half-edges).
    pub fn chain_vertices(&self, chain: &[HalfEdgeId]) -> Vec<VertexId> {
        let mut vertices: Vec<VertexId> = chain.iter().map(|&h| self.origin(h)).collect();
        if let Some(&last) = chain.last()
            && !self.is_closed_chain(chain)
        {
            vertices.push(self.target(last));
        }
        vertices
    }

    /// Extrudes border half-edges: each vertex on them gets a copy at
    /// `position(vertex)`, and each half-edge a quad joining it to the
    /// copies. Returns the copy of each vertex.
    pub fn extrude_border(
        &mut self,
        halfedges: &[HalfEdgeId],
        position: impl Fn(VertexId) -> Vec3,
    ) -> Result<HashMap<VertexId, VertexId>, TopologyError> {
        for &h in halfedges {
            self.check_halfedge(h)?;
            if !self.is_boundary_halfedge(h) {
                return Err(TopologyError::Refused("only border edges can be extruded"));
            }
        }
        let mut copies: HashMap<VertexId, VertexId> = HashMap::new();
        let mut faces = Vec::new();
        for &h in halfedges {
            let (a, b) = (self.origin(h), self.target(h));
            for v in [a, b] {
                if let std::collections::hash_map::Entry::Vacant(slot) = copies.entry(v) {
                    slot.insert(self.add_vertex(position(v)));
                }
            }
            faces.push(corners_of(&[a, b, copies[&b], copies[&a]]));
        }
        self.rebuild(&[], faces)?;
        Ok(copies)
    }

    /// Joins two chains of border half-edges (see [`Self::border_chains`])
    /// with a row of quads. They need as many edges, and must be both open
    /// or both closed; closed chains are lined up where their vertices are
    /// closest. Chains sharing an end get a triangle there.
    pub fn bridge(&mut self, a: &[HalfEdgeId], b: &[HalfEdgeId]) -> Result<(), TopologyError> {
        for &h in a.iter().chain(b) {
            self.check_halfedge(h)?;
            if !self.is_boundary_halfedge(h) {
                return Err(TopologyError::Refused("only border edges can be bridged"));
            }
        }
        let n = a.len();
        if n == 0 || n != b.len() {
            return Err(TopologyError::Refused(
                "the two borders need the same number of edges",
            ));
        }
        let closed = self.is_closed_chain(a);
        if closed != self.is_closed_chain(b) {
            return Err(TopologyError::Refused(
                "a closed border can only be bridged to another closed one",
            ));
        }
        let (va, vb) = (self.chain_vertices(a), self.chain_vertices(b));
        // Quad i runs along a[i] and back along the matching edge of b; b is
        // walked backwards, since facing borders run opposite ways.
        let offset = if closed {
            let cost = |o: usize| -> f32 {
                (0..n)
                    .map(|i| {
                        self.position(va[i])
                            .distance_squared(self.position(vb[(o + n - i) % n]))
                    })
                    .sum()
            };
            (0..n)
                .min_by(|&x, &y| cost(x).total_cmp(&cost(y)))
                .expect("n > 0")
        } else {
            n
        };
        let m = vb.len();
        let mut faces = Vec::with_capacity(n);
        for i in 0..n {
            let (a0, a1) = (va[i], va[(i + 1) % va.len()]);
            let (b0, b1) = (vb[(offset + 2 * m - i - 1) % m], vb[(offset + m - i) % m]);
            let mut ring: Vec<VertexId> = vec![a0, a1, b0, b1];
            ring.dedup();
            if ring.first() == ring.last() {
                ring.pop();
            }
            if ring.len() >= 3 {
                faces.push(corners_of(&ring));
            }
        }
        self.rebuild(&[], faces)?;
        Ok(())
    }

    /// Ways to fill the hole of border half-edge `h` with a grid of quads,
    /// best first: the four corners, which split the border into opposite
    /// sides of equal length. Corners go where the border turns most, and
    /// the grid is as square as the border allows. None for a hole with an
    /// odd number of edges.
    pub fn grid_fill_corners(&self, h: HalfEdgeId) -> Vec<[VertexId; 4]> {
        if !self.contains_halfedge(h) || !self.is_boundary_halfedge(h) {
            return Vec::new();
        }
        let ring: Vec<VertexId> = self.loop_from(h).map(|h| self.origin(h)).collect();
        let n = ring.len();
        if n < 4 || n % 2 == 1 {
            return Vec::new();
        }
        let turns: Vec<f32> = (0..n)
            .map(|k| {
                let p = |i: usize| self.position(ring[i % n]);
                let (before, after) = (p(k + n) - p(k + n - 1), p(k + 1) - p(k));
                before.angle_between(after)
            })
            .map(|t| if t.is_finite() { t } else { 0.0 })
            .collect();
        // Cost of a vertex as a corner, relative to being on a side.
        let corner_cost: Vec<f32> = turns
            .iter()
            .map(|&t| (t - FRAC_PI_2).powi(2) - t * t)
            .collect();
        let half = n / 2;
        let mut candidates: Vec<(f32, [VertexId; 4])> = Vec::new();
        for start in 0..half {
            for a in 1..half {
                let b = half - a;
                let idx = [start, start + a, start + a + b, start + 2 * a + b].map(|i| i % n);
                let squareness = ((a as f32 - b as f32) / half as f32).powi(2) * 0.1;
                let cost = idx.iter().map(|&i| corner_cost[i]).sum::<f32>() + squareness;
                candidates.push((cost, idx.map(|i| ring[i])));
            }
        }
        candidates.sort_by(|x, y| x.0.total_cmp(&y.0));
        candidates.into_iter().map(|(_, c)| c).collect()
    }

    /// Fills the hole of border half-edge `h`: with a grid of quads between
    /// `corners` (see [`Self::grid_fill_corners`]; in any order), or with
    /// one polygon.
    /// Inner grid vertices are placed by blending the sides (a Coons patch);
    /// returns them.
    pub fn fill_hole(
        &mut self,
        h: HalfEdgeId,
        corners: Option<[VertexId; 4]>,
    ) -> Result<Vec<VertexId>, TopologyError> {
        self.check_halfedge(h)?;
        if !self.is_boundary_halfedge(h) {
            return Err(TopologyError::Refused("the edge is not on a border"));
        }
        let ring: Vec<VertexId> = self.loop_from(h).map(|h| self.origin(h)).collect();
        let n = ring.len();
        let Some(corners) = corners else {
            if n < 3 {
                return Err(TopologyError::TooFewVertices);
            }
            self.rebuild(&[], vec![corners_of(&ring)])?;
            return Ok(Vec::new());
        };
        let index = |v| ring.iter().position(|&r| r == v);
        let Some(mut idx) = corners
            .iter()
            .map(|&c| index(c))
            .collect::<Option<Vec<usize>>>()
        else {
            return Err(TopologyError::Refused(
                "the corners are not on the hole's border",
            ));
        };
        // In border order, whatever order they were given in.
        idx.sort_unstable();
        let side = |i: usize| (idx[(i + 1) % 4] + n - idx[i]) % n;
        let (a, b) = (side(0), side(1));
        if a == 0 || b == 0 || side(2) != a || side(3) != b {
            return Err(TopologyError::Refused(
                "opposite sides of a grid fill need as many edges",
            ));
        }
        let s = idx[0];
        let at = |k: usize| ring[(s + k) % n];
        // Border vertex of grid point (i, j), going around from the start.
        let border = |i: usize, j: usize| -> Option<VertexId> {
            match (i, j) {
                (_, 0) => Some(at(i)),
                (i, _) if i == a => Some(at(a + j)),
                (_, j) if j == b => Some(at(a + b + (a - i))),
                (0, _) => Some(at(2 * a + b + (b - j))),
                _ => None,
            }
        };
        let p = |v: VertexId| self.position(v);
        let corner_points = [p(at(0)), p(at(a)), p(at(a + b)), p(at(2 * a + b))];
        // Inner points first (the mesh is read), then their vertices.
        let mut cells: Vec<Vec<Result<VertexId, Vec3>>> = Vec::with_capacity(a + 1);
        for i in 0..=a {
            let column = (0..=b)
                .map(|j| match border(i, j) {
                    Some(v) => Ok(v),
                    None => {
                        let (u, w) = (i as f32 / a as f32, j as f32 / b as f32);
                        let bottom = p(at(i));
                        let top = p(at(a + b + (a - i)));
                        let left = p(at(2 * a + b + (b - j)));
                        let right = p(at(a + j));
                        let [c0, c1, c2, c3] = corner_points;
                        let bilinear = c0 * (1.0 - u) * (1.0 - w)
                            + c1 * u * (1.0 - w)
                            + c2 * u * w
                            + c3 * (1.0 - u) * w;
                        Err(bottom * (1.0 - w) + top * w + left * (1.0 - u) + right * u - bilinear)
                    }
                })
                .collect();
            cells.push(column);
        }
        let mut inner = Vec::new();
        let grid: Vec<Vec<VertexId>> = cells
            .into_iter()
            .map(|column| {
                column
                    .into_iter()
                    .map(|cell| match cell {
                        Ok(v) => v,
                        Err(position) => {
                            let v = self.add_vertex(position);
                            inner.push(v);
                            v
                        }
                    })
                    .collect()
            })
            .collect();
        let mut faces = Vec::with_capacity(a * b);
        for i in 0..a {
            for j in 0..b {
                faces.push(corners_of(&[
                    grid[i][j],
                    grid[i + 1][j],
                    grid[i + 1][j + 1],
                    grid[i][j + 1],
                ]));
            }
        }
        self.rebuild(&[], faces)?;
        Ok(inner)
    }
}

fn corners_of(vertices: &[VertexId]) -> Vec<Corner> {
    vertices
        .iter()
        .map(|&vertex| Corner { vertex, uv: None })
        .collect()
}
