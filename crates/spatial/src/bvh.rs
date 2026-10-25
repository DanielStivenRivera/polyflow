//! Bounding volume hierarchy over the triangles of a mesh.
//!
//! Built top-down with a binned surface area heuristic. The upper levels are
//! built in parallel; each parallel task produces a position-independent node
//! array (children are addressed relative to their parent), and the pieces
//! are copied once into the final array.

use bytemuck::{Pod, Zeroable};
use polyflow_core::bounds::Aabb;
use polyflow_core::glam::Vec3;
use polyflow_core::mesh::TriMesh;
use rayon::prelude::*;

use crate::triangle;

const BINS: usize = 16;
/// Leaves never hold more triangles than this.
const MAX_LEAF: usize = 8;
/// Relative cost of visiting a node versus intersecting a triangle.
/// Values above 1 favor fewer, fuller leaves: 4 uses 60% less memory than 1
/// with equal query speed on sculpt-like meshes.
const TRAVERSAL_COST: f32 = 4.0;
/// Ranges at least this large are split in parallel.
const PARALLEL_MIN: usize = 1 << 15;
/// Ranges at least this large are binned in parallel.
const PARALLEL_BINNING_MIN: usize = 1 << 17;
/// Below this depth splits use the SAH; deeper ones split at the median so
/// the depth (and the traversal stack) stays bounded for any input.
const MAX_SAH_DEPTH: usize = 48;
/// Stack size of the build threads.
const BUILD_STACK_BYTES: usize = 64 << 20;
/// Traversal stack depth that always suffices: `MAX_SAH_DEPTH` plus `log2`
/// of the largest possible mesh.
pub const STACK_SIZE: usize = MAX_SAH_DEPTH + 40;

/// A node in depth-first order. Internal nodes have their left child right
/// after them and their right child at `self + index`; leaves (`count > 0`)
/// own `triangles[index..index + count]`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct Node {
    min: Vec3,
    index: u32,
    max: Vec3,
    count: u32,
}

impl Node {
    #[cfg(test)]
    fn bounds(&self) -> Aabb {
        Aabb {
            min: self.min,
            max: self.max,
        }
    }

    fn leaf(bounds: Aabb, first: usize, count: usize) -> Self {
        Self {
            min: bounds.min,
            max: bounds.max,
            index: first as u32,
            count: count as u32,
        }
    }

    fn internal(bounds: Aabb, right_offset: usize) -> Self {
        Self {
            min: bounds.min,
            max: bounds.max,
            index: right_offset as u32,
            count: 0,
        }
    }
}

/// A node of a [`Bvh`], as stored by [`Bvh::parts`].
pub type BvhNode = Node;

/// A BVH over a [`TriMesh`]. Queries take the mesh it was built from.
#[derive(Debug, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    /// Triangle ids in leaf order.
    triangles: Vec<u32>,
}

/// A ray hit. The hit point is `(1 - u - v) * a + u * b + v * c` for the
/// triangle's vertices `a`, `b`, `c`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub t: f32,
    pub triangle: u32,
    pub u: f32,
    pub v: f32,
}

/// The point of the mesh closest to a query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosestPoint {
    pub point: Vec3,
    pub distance: f32,
    pub triangle: u32,
    pub u: f32,
    pub v: f32,
}

impl Bvh {
    /// Builds the hierarchy using all cores.
    ///
    /// # Panics
    /// If the mesh has more than `u32::MAX` triangles or an index is out of
    /// bounds; [`TriMesh`] values from the importers never do.
    pub fn build(mesh: &TriMesh) -> Self {
        // Threads blocked in `rayon::join` run stolen subtree builds on top of
        // their own stack, so the default 2 MB stacks can overflow on huge
        // meshes. Build on a dedicated pool with roomy stacks.
        match rayon::ThreadPoolBuilder::new()
            .stack_size(BUILD_STACK_BYTES)
            .thread_name(|i| format!("bvh build {i}"))
            .build()
        {
            Ok(pool) => pool.install(|| Self::build_in_current_pool(mesh)),
            Err(e) => {
                log::warn!(
                    "could not create the BVH build thread pool ({e}); using the global pool"
                );
                Self::build_in_current_pool(mesh)
            }
        }
    }

    fn build_in_current_pool(mesh: &TriMesh) -> Self {
        let triangle_count = mesh.triangle_count();
        if triangle_count == 0 {
            return Self::default();
        }
        assert!(u32::try_from(triangle_count).is_ok(), "too many triangles");
        let ctx = BuildContext {
            mesh,
            centroids: mesh
                .indices
                .par_chunks_exact(3)
                .map(|t| {
                    (mesh.positions[t[0] as usize]
                        + mesh.positions[t[1] as usize]
                        + mesh.positions[t[2] as usize])
                        / 3.0
                })
                .collect(),
        };
        let mut triangles: Vec<u32> = (0..triangle_count as u32).collect();
        let bounds = ctx.bounds_of(&triangles);
        let tree = build_parallel(&ctx, &mut triangles, 0, bounds, 0);
        let mut nodes = vec![Node::leaf(Aabb::EMPTY, 0, 0); tree.len()];
        tree.write_into(&mut nodes);
        Self { nodes, triangles }
    }

    /// The nodes and the triangle order, to store and rebuild the hierarchy
    /// with [`Self::from_parts`].
    pub fn parts(&self) -> (&[BvhNode], &[u32]) {
        (&self.nodes, &self.triangles)
    }

    /// Rebuilds a hierarchy from [`Self::parts`] (e.g. read from a cache),
    /// checking that it is well formed for a mesh of `triangle_count`
    /// triangles, so queries on it cannot go out of bounds or overflow
    /// their stack.
    pub fn from_parts(
        nodes: Vec<BvhNode>,
        triangles: Vec<u32>,
        triangle_count: usize,
    ) -> Result<Self, String> {
        if triangles.len() != triangle_count {
            return Err("the triangle order does not match the mesh".into());
        }
        if triangles.iter().any(|&t| t as usize >= triangle_count) {
            return Err("a triangle id is out of range".into());
        }
        if nodes.is_empty() != (triangle_count == 0) {
            return Err("the hierarchy does not match the mesh".into());
        }
        // Children come after their parent, so one forward pass sees every
        // node's depth before its children.
        let mut depth = vec![0u32; nodes.len()];
        if let Some(d) = depth.first_mut() {
            *d = 1;
        }
        for (i, node) in nodes.iter().enumerate() {
            let d = depth[i];
            if d == 0 {
                return Err(format!("node {i} is not reachable"));
            }
            if d as usize > STACK_SIZE {
                return Err("the hierarchy is too deep".into());
            }
            if node.count > 0 {
                let end = node.index as usize + node.count as usize;
                if end > triangles.len() {
                    return Err(format!("leaf {i} runs past the triangles"));
                }
                continue;
            }
            let (left, right) = (i + 1, i + node.index as usize);
            if node.index < 2 || right >= nodes.len() {
                return Err(format!("node {i} has invalid children"));
            }
            for child in [left, right] {
                if depth[child] != 0 {
                    return Err(format!("node {child} has two parents"));
                }
                depth[child] = d + 1;
            }
        }
        Ok(Self { nodes, triangles })
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The hierarchy as plain words, for GPU traversal: 8 per node, in
    /// storage order: `min` x, y, z (`f32` bits), `first`, `max` x, y, z,
    /// `count`. A leaf has `count > 0` triangles,
    /// `triangle_order()[first..first + count]`. An interior node has
    /// `count == 0`; its left child follows it and its right child is
    /// `first` nodes after it.
    pub fn gpu_nodes(&self) -> Vec<u32> {
        self.nodes
            .iter()
            .flat_map(|n| {
                let [a, b, c] = n.min.to_array().map(f32::to_bits);
                let [d, e, f] = n.max.to_array().map(f32::to_bits);
                [a, b, c, n.index, d, e, f, n.count]
            })
            .collect()
    }

    /// Triangle ids in leaf order (see [`Self::gpu_nodes`]).
    pub fn triangle_order(&self) -> &[u32] {
        &self.triangles
    }

    /// Heap memory used by the hierarchy.
    pub fn memory_bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<Node>() + self.triangles.capacity() * size_of::<u32>()
    }

    /// Nearest intersection of the ray `origin + t * dir` with `0 <= t <= t_max`.
    /// Both sides of triangles are hit. `dir` need not be normalized; `t` is
    /// in units of `dir`.
    pub fn raycast(&self, mesh: &TriMesh, origin: Vec3, dir: Vec3, t_max: f32) -> Option<RayHit> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv_dir = dir.recip();
        let mut best: Option<RayHit> = None;
        let mut best_t = t_max;
        let mut stack = [0u32; STACK_SIZE];
        let mut len = 0;
        if ray_box(origin, inv_dir, &self.nodes[0], best_t).is_some() {
            stack[0] = 0;
            len = 1;
        }
        while len > 0 {
            len -= 1;
            let index = stack[len] as usize;
            let node = &self.nodes[index];
            if node.count > 0 {
                for &tri in self.leaf_triangles(node) {
                    let [a, b, c] = triangle_vertices(mesh, tri);
                    if let Some((t, u, v)) = triangle::intersect_ray(origin, dir, a, b, c)
                        && t >= 0.0
                        && t <= best_t
                    {
                        best_t = t;
                        best = Some(RayHit {
                            t,
                            triangle: tri,
                            u,
                            v,
                        });
                    }
                }
                continue;
            }
            let (left, right) = (index + 1, index + node.index as usize);
            let hit_left = ray_box(origin, inv_dir, &self.nodes[left], best_t);
            let hit_right = ray_box(origin, inv_dir, &self.nodes[right], best_t);
            // Push the farther child first so the nearer one is visited first.
            match (hit_left, hit_right) {
                (Some(l), Some(r)) => {
                    let (near, far) = if l <= r { (left, right) } else { (right, left) };
                    stack[len] = far as u32;
                    stack[len + 1] = near as u32;
                    len += 2;
                }
                (Some(_), None) => {
                    stack[len] = left as u32;
                    len += 1;
                }
                (None, Some(_)) => {
                    stack[len] = right as u32;
                    len += 1;
                }
                (None, None) => {}
            }
        }
        best
    }

    /// Whether the ray `origin + t * dir`, `0 <= t <= t_max`, hits anything.
    /// Stops at the first hit found, so it is cheaper than [`Self::raycast`]
    /// for visibility (ambient occlusion).
    pub fn occluded(&self, mesh: &TriMesh, origin: Vec3, dir: Vec3, t_max: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv_dir = dir.recip();
        let mut stack = [0u32; STACK_SIZE];
        let mut len = 0;
        if ray_box(origin, inv_dir, &self.nodes[0], t_max).is_some() {
            stack[0] = 0;
            len = 1;
        }
        while len > 0 {
            len -= 1;
            let index = stack[len] as usize;
            let node = &self.nodes[index];
            if node.count > 0 {
                for &tri in self.leaf_triangles(node) {
                    let [a, b, c] = triangle_vertices(mesh, tri);
                    if let Some((t, _, _)) = triangle::intersect_ray(origin, dir, a, b, c)
                        && t >= 0.0
                        && t <= t_max
                    {
                        return true;
                    }
                }
                continue;
            }
            for child in [index + 1, index + node.index as usize] {
                if ray_box(origin, inv_dir, &self.nodes[child], t_max).is_some() {
                    stack[len] = child as u32;
                    len += 1;
                }
            }
        }
        false
    }

    /// Point of the mesh closest to `p`, if any is within `max_distance`.
    pub fn closest_point(
        &self,
        mesh: &TriMesh,
        p: Vec3,
        max_distance: f32,
    ) -> Option<ClosestPoint> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best: Option<ClosestPoint> = None;
        let mut best_d2 = max_distance * max_distance;
        let mut stack = [0u32; STACK_SIZE];
        let mut len = 0;
        if box_distance_squared(p, &self.nodes[0]) <= best_d2 {
            stack[0] = 0;
            len = 1;
        }
        while len > 0 {
            len -= 1;
            let index = stack[len] as usize;
            let node = &self.nodes[index];
            // The bound may have tightened since this node was pushed.
            if box_distance_squared(p, node) > best_d2 {
                continue;
            }
            if node.count > 0 {
                for &tri in self.leaf_triangles(node) {
                    let [a, b, c] = triangle_vertices(mesh, tri);
                    let (q, u, v) = triangle::closest_point(p, a, b, c);
                    let d2 = q.distance_squared(p);
                    if d2 <= best_d2 {
                        best_d2 = d2;
                        best = Some(ClosestPoint {
                            point: q,
                            distance: d2.sqrt(),
                            triangle: tri,
                            u,
                            v,
                        });
                    }
                }
                continue;
            }
            let (left, right) = (index + 1, index + node.index as usize);
            let dl = box_distance_squared(p, &self.nodes[left]);
            let dr = box_distance_squared(p, &self.nodes[right]);
            let (near, d_near, far, d_far) = if dl <= dr {
                (left, dl, right, dr)
            } else {
                (right, dr, left, dl)
            };
            if d_far <= best_d2 {
                stack[len] = far as u32;
                len += 1;
            }
            if d_near <= best_d2 {
                stack[len] = near as u32;
                len += 1;
            }
        }
        best
    }

    fn leaf_triangles(&self, node: &Node) -> &[u32] {
        &self.triangles[node.index as usize..(node.index + node.count) as usize]
    }

    /// Checks structural invariants. For tests.
    #[cfg(test)]
    fn validate(&self, mesh: &TriMesh) {
        let mut seen = vec![false; mesh.triangle_count()];
        let mut max_depth = 0;
        let mut stack = vec![(0usize, 0usize)];
        while let Some((index, depth)) = stack.pop() {
            max_depth = max_depth.max(depth);
            let node = &self.nodes[index];
            if node.count > 0 {
                assert!(node.count as usize <= MAX_LEAF.max(1));
                for &tri in self.leaf_triangles(node) {
                    assert!(
                        !std::mem::replace(&mut seen[tri as usize], true),
                        "triangle in two leaves"
                    );
                    for v in triangle_vertices(mesh, tri) {
                        assert!(
                            node.bounds().grow(v) == node.bounds(),
                            "triangle outside its leaf"
                        );
                    }
                }
            } else {
                for child in [index + 1, index + node.index as usize] {
                    let b = self.nodes[child].bounds();
                    assert!(
                        node.bounds().union(b) == node.bounds(),
                        "child outside its parent"
                    );
                    stack.push((child, depth + 1));
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "every triangle is in a leaf");
        assert!(
            max_depth < STACK_SIZE,
            "depth {max_depth} exceeds the traversal stack"
        );
    }
}

fn triangle_vertices(mesh: &TriMesh, tri: u32) -> [Vec3; 3] {
    let i = tri as usize * 3;
    [0, 1, 2].map(|k| mesh.positions[mesh.indices[i + k] as usize])
}

/// Entry distance of the ray into the node's box, if it enters within `[0, t_max]`.
#[inline]
fn ray_box(origin: Vec3, inv_dir: Vec3, node: &Node, t_max: f32) -> Option<f32> {
    let t0 = (node.min - origin) * inv_dir;
    let t1 = (node.max - origin) * inv_dir;
    // `min`/`max` drop the NaN produced by 0 * inf when the origin lies on a slab plane.
    let near = t0.min(t1).max_element().max(0.0);
    let far = t0.max(t1).min_element().min(t_max);
    (near <= far).then_some(near)
}

#[inline]
fn box_distance_squared(p: Vec3, node: &Node) -> f32 {
    let d = (node.min - p).max(p - node.max).max(Vec3::ZERO);
    d.length_squared()
}

struct BuildContext<'a> {
    mesh: &'a TriMesh,
    centroids: Vec<Vec3>,
}

impl BuildContext<'_> {
    fn triangle_bounds(&self, tri: u32) -> Aabb {
        let [a, b, c] = triangle_vertices(self.mesh, tri);
        Aabb {
            min: a.min(b).min(c),
            max: a.max(b).max(c),
        }
    }

    fn bounds_of(&self, tris: &[u32]) -> Aabb {
        let fold = |acc: Aabb, &t: &u32| acc.union(self.triangle_bounds(t));
        if tris.len() >= PARALLEL_BINNING_MIN {
            tris.par_iter()
                .fold(|| Aabb::EMPTY, fold)
                .reduce(|| Aabb::EMPTY, Aabb::union)
        } else {
            tris.iter().fold(Aabb::EMPTY, fold)
        }
    }

    fn centroid_bounds(&self, tris: &[u32]) -> Aabb {
        let fold = |acc: Aabb, &t: &u32| acc.grow(self.centroids[t as usize]);
        if tris.len() >= PARALLEL_BINNING_MIN {
            tris.par_iter()
                .fold(|| Aabb::EMPTY, fold)
                .reduce(|| Aabb::EMPTY, Aabb::union)
        } else {
            tris.iter().fold(Aabb::EMPTY, fold)
        }
    }
}

/// A subtree whose upper levels were split across threads.
enum Subtree {
    Built(Vec<Node>),
    Split {
        bounds: Aabb,
        left: Box<Subtree>,
        right: Box<Subtree>,
        len: usize,
    },
}

impl Subtree {
    fn len(&self) -> usize {
        match self {
            Subtree::Built(nodes) => nodes.len(),
            Subtree::Split { len, .. } => *len,
        }
    }

    /// Writes the subtree into `out`, which has exactly `self.len()` nodes.
    fn write_into(self, out: &mut [Node]) {
        match self {
            Subtree::Built(nodes) => out.copy_from_slice(&nodes),
            Subtree::Split {
                bounds,
                left,
                right,
                ..
            } => {
                let left_len = left.len();
                out[0] = Node::internal(bounds, 1 + left_len);
                let (left_out, right_out) = out[1..].split_at_mut(left_len);
                rayon::join(|| left.write_into(left_out), || right.write_into(right_out));
            }
        }
    }
}

fn build_parallel(
    ctx: &BuildContext<'_>,
    tris: &mut [u32],
    first: usize,
    bounds: Aabb,
    depth: usize,
) -> Subtree {
    if tris.len() < PARALLEL_MIN {
        let mut nodes = Vec::with_capacity(tris.len() / 2);
        build_sequential(ctx, tris, first, bounds, depth, &mut nodes);
        return Subtree::Built(nodes);
    }
    let Some(split) = choose_split(ctx, tris, bounds, depth) else {
        // Unreachable for ranges this large (leaves are capped), but keep a valid tree.
        return Subtree::Built(vec![Node::leaf(bounds, first, tris.len())]);
    };
    let (left_tris, right_tris) = tris.split_at_mut(split.left_count);
    let (left, right) = rayon::join(
        || build_parallel(ctx, left_tris, first, split.left_bounds, depth + 1),
        || {
            build_parallel(
                ctx,
                right_tris,
                first + split.left_count,
                split.right_bounds,
                depth + 1,
            )
        },
    );
    let len = 1 + left.len() + right.len();
    Subtree::Split {
        bounds,
        left: Box::new(left),
        right: Box::new(right),
        len,
    }
}

fn build_sequential(
    ctx: &BuildContext<'_>,
    tris: &mut [u32],
    first: usize,
    bounds: Aabb,
    depth: usize,
    out: &mut Vec<Node>,
) {
    let me = out.len();
    let Some(split) = choose_split(ctx, tris, bounds, depth) else {
        out.push(Node::leaf(bounds, first, tris.len()));
        return;
    };
    out.push(Node::internal(bounds, 0)); // Right offset patched below.
    let (left, right) = tris.split_at_mut(split.left_count);
    build_sequential(ctx, left, first, split.left_bounds, depth + 1, out);
    let right_offset = out.len() - me;
    build_sequential(
        ctx,
        right,
        first + split.left_count,
        split.right_bounds,
        depth + 1,
        out,
    );
    out[me].index = right_offset as u32;
}

struct Split {
    left_count: usize,
    left_bounds: Aabb,
    right_bounds: Aabb,
}

#[derive(Clone, Copy)]
struct Bin {
    bounds: Aabb,
    count: usize,
}

const EMPTY_BIN: Bin = Bin {
    bounds: Aabb::EMPTY,
    count: 0,
};

/// Decides how to split `tris` (reordering them so the left part comes
/// first), or returns `None` to make a leaf.
fn choose_split(
    ctx: &BuildContext<'_>,
    tris: &mut [u32],
    bounds: Aabb,
    depth: usize,
) -> Option<Split> {
    let n = tris.len();
    if n <= 2 {
        return None;
    }
    let centroid_bounds = ctx.centroid_bounds(tris);
    let extent = centroid_bounds.max - centroid_bounds.min;
    if depth >= MAX_SAH_DEPTH || extent.max_element() <= 0.0 {
        // Coincident centroids or a pathologically deep branch.
        return (n > MAX_LEAF).then(|| median_split(ctx, tris));
    }

    // Bin along the longest centroid axis only: a third of the work of
    // binning all three, for nearly the same tree quality on scanned and
    // sculpted meshes.
    let axis = extent.max_position();
    let scale = BINS as f32 / extent[axis];
    let min = centroid_bounds.min[axis];
    let bin_of =
        |t: u32| (((ctx.centroids[t as usize][axis] - min) * scale) as usize).min(BINS - 1);
    let add = |mut bins: [Bin; BINS], &t: &u32| {
        let bin = &mut bins[bin_of(t)];
        bin.bounds = bin.bounds.union(ctx.triangle_bounds(t));
        bin.count += 1;
        bins
    };
    let merge = |mut a: [Bin; BINS], b: [Bin; BINS]| {
        for (a, b) in a.iter_mut().zip(b) {
            a.bounds = a.bounds.union(b.bounds);
            a.count += b.count;
        }
        a
    };
    let empty = [EMPTY_BIN; BINS];
    let bins = if n >= PARALLEL_BINNING_MIN {
        tris.par_iter().fold(|| empty, add).reduce(|| empty, merge)
    } else {
        tris.iter().fold(empty, add)
    };

    // Sweep for the cheapest split between bins.
    let parent_area = surface_area(&bounds).max(f32::MIN_POSITIVE);
    let mut right_bounds = [Aabb::EMPTY; BINS];
    let mut acc = Aabb::EMPTY;
    for i in (1..BINS).rev() {
        acc = acc.union(bins[i].bounds);
        right_bounds[i] = acc;
    }
    let mut best: Option<(f32, usize, Aabb, Aabb, usize)> = None; // cost, split bin, left, right, left count
    let (mut left, mut left_count) = (Aabb::EMPTY, 0);
    for i in 1..BINS {
        left = left.union(bins[i - 1].bounds);
        left_count += bins[i - 1].count;
        let right_count = n - left_count;
        if left_count == 0 || right_count == 0 {
            continue;
        }
        let cost = TRAVERSAL_COST
            + (surface_area(&left) * left_count as f32
                + surface_area(&right_bounds[i]) * right_count as f32)
                / parent_area;
        if best.is_none_or(|b| cost < b.0) {
            best = Some((cost, i, left, right_bounds[i], left_count));
        }
    }

    let Some((cost, split_bin, left_bounds, right_bounds, left_count)) = best else {
        return (n > MAX_LEAF).then(|| median_split(ctx, tris));
    };
    if n <= MAX_LEAF && n as f32 <= cost {
        return None;
    }
    partition(tris, |t| bin_of(t) < split_bin);
    Some(Split {
        left_count,
        left_bounds,
        right_bounds,
    })
}

/// Splits at the middle along the longest axis of the centroids.
fn median_split(ctx: &BuildContext<'_>, tris: &mut [u32]) -> Split {
    let extent = {
        let b = ctx.centroid_bounds(tris);
        b.max - b.min
    };
    let axis = extent.max_position();
    let mid = tris.len() / 2;
    tris.select_nth_unstable_by(mid, |&a, &b| {
        ctx.centroids[a as usize][axis].total_cmp(&ctx.centroids[b as usize][axis])
    });
    let (left, right) = tris.split_at(mid);
    Split {
        left_count: mid,
        left_bounds: ctx.bounds_of(left),
        right_bounds: ctx.bounds_of(right),
    }
}

/// Moves elements matching `left` to the front, in place.
fn partition(tris: &mut [u32], left: impl Fn(u32) -> bool) {
    let mut first_right = 0;
    for i in 0..tris.len() {
        if left(tris[i]) {
            tris.swap(i, first_right);
            first_right += 1;
        }
    }
}

/// Half the surface area of the box (the factor cancels in the SAH).
fn surface_area(b: &Aabb) -> f32 {
    if b.is_empty() {
        return 0.0;
    }
    let d = b.max - b.min;
    d.x * d.y + d.y * d.z + d.z * d.x
}

#[cfg(test)]
mod tests {
    use polyflow_core::primitives::torus;

    use super::*;

    /// Small deterministic PRNG, so tests need no extra dependency.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn vec3(&mut self, scale: f32) -> Vec3 {
            Vec3::new(self.next_f32(), self.next_f32(), self.next_f32()) * 2.0 * scale - scale
        }
    }

    #[test]
    fn parts_round_trip_and_are_checked() {
        let mesh = torus(1.0, 0.3, 64, 32);
        let bvh = Bvh::build(&mesh);
        let (nodes, triangles) = bvh.parts();
        let count = mesh.triangle_count();
        let back = Bvh::from_parts(nodes.to_vec(), triangles.to_vec(), count).unwrap();
        let (origin, dir) = (Vec3::new(3.0, 0.1, 0.2), Vec3::NEG_X);
        assert_eq!(
            back.raycast(&mesh, origin, dir, f32::INFINITY),
            bvh.raycast(&mesh, origin, dir, f32::INFINITY)
        );

        let broken = |edit: &dyn Fn(&mut Vec<BvhNode>, &mut Vec<u32>)| {
            let (mut n, mut t) = (nodes.to_vec(), triangles.to_vec());
            edit(&mut n, &mut t);
            Bvh::from_parts(n, t, count)
        };
        assert!(broken(&|_, t| t.truncate(10)).is_err());
        assert!(broken(&|_, t| t[3] = count as u32).is_err());
        assert!(broken(&|n, _| n[0].index = 1).is_err());
        assert!(broken(&|n, _| n[0].index = u32::MAX).is_err());
        let leaf = nodes.iter().position(|n| n.count > 0).unwrap();
        assert!(broken(&|n, _| n[leaf].count = u32::MAX).is_err());
        // A tree deeper than the traversal stack.
        let len = 2 * STACK_SIZE + 3;
        let chain: Vec<BvhNode> = (0..len)
            .map(|i| {
                if i % 2 == 0 && i + 2 < len {
                    Node::internal(Aabb::EMPTY, 2)
                } else {
                    Node::leaf(Aabb::EMPTY, 0, 1)
                }
            })
            .collect();
        assert!(Bvh::from_parts(chain, vec![0], 1).is_err());
        assert!(Bvh::from_parts(Vec::new(), Vec::new(), 0).is_ok());
    }

    fn triangle_soup(count: usize, rng: &mut Rng) -> TriMesh {
        let mut mesh = TriMesh::default();
        for i in 0..count {
            let center = rng.vec3(5.0);
            for _ in 0..3 {
                mesh.positions.push(center + rng.vec3(0.4));
            }
            mesh.indices
                .extend_from_slice(&[3 * i as u32, 3 * i as u32 + 1, 3 * i as u32 + 2]);
        }
        mesh.compute_normals();
        mesh
    }

    fn brute_raycast(mesh: &TriMesh, origin: Vec3, dir: Vec3) -> Option<(f32, u32)> {
        let mut best: Option<(f32, u32)> = None;
        for tri in 0..mesh.triangle_count() as u32 {
            let [a, b, c] = triangle_vertices(mesh, tri);
            if let Some((t, ..)) = triangle::intersect_ray(origin, dir, a, b, c)
                && t >= 0.0
                && best.is_none_or(|(bt, _)| t < bt)
            {
                best = Some((t, tri));
            }
        }
        best
    }

    fn brute_closest(mesh: &TriMesh, p: Vec3) -> f32 {
        (0..mesh.triangle_count() as u32)
            .map(|tri| {
                let [a, b, c] = triangle_vertices(mesh, tri);
                triangle::closest_point(p, a, b, c).0.distance(p)
            })
            .fold(f32::INFINITY, f32::min)
    }

    fn check_against_brute_force(mesh: &TriMesh, rng: &mut Rng) {
        let bvh = Bvh::build(mesh);
        bvh.validate(mesh);
        for _ in 0..300 {
            let origin = rng.vec3(8.0);
            let dir = (rng.vec3(1.0) - origin * 0.1).normalize_or(Vec3::X);
            let expected = brute_raycast(mesh, origin, dir);
            let actual = bvh.raycast(mesh, origin, dir, f32::INFINITY);
            match (expected, actual) {
                (None, None) => {}
                (Some((t, _)), Some(hit)) => {
                    assert!((hit.t - t).abs() <= 1e-4 * t.max(1.0), "{t} vs {}", hit.t)
                }
                other => panic!("raycast mismatch: {other:?}"),
            }
            // Occlusion agrees on segments ending just past or short of the
            // nearest hit.
            match expected {
                Some((t, _)) => {
                    assert!(bvh.occluded(mesh, origin, dir, t * 1.001 + 1e-4));
                    if t > 1e-2 {
                        assert!(!bvh.occluded(mesh, origin, dir, t * 0.999 - 1e-4));
                    }
                }
                None => assert!(!bvh.occluded(mesh, origin, dir, f32::INFINITY)),
            }

            let p = rng.vec3(7.0);
            let expected = brute_closest(mesh, p);
            let actual = bvh.closest_point(mesh, p, f32::INFINITY).unwrap();
            assert!(
                (actual.distance - expected).abs() <= 1e-4,
                "{expected} vs {}",
                actual.distance
            );
            assert!((actual.point.distance(p) - actual.distance).abs() <= 1e-4);
        }
    }

    #[test]
    fn matches_brute_force_on_triangle_soup() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mesh = triangle_soup(3000, &mut rng);
        check_against_brute_force(&mesh, &mut rng);
    }

    #[test]
    fn matches_brute_force_on_torus() {
        let mut rng = Rng(42);
        let mut mesh = torus(4.0, 1.2, 96, 48);
        check_against_brute_force(&mesh, &mut rng);
        // Also exercise the parallel build path on a larger mesh.
        mesh = torus(4.0, 1.2, 512, 256);
        check_against_brute_force(&mesh, &mut rng);
    }

    #[test]
    fn handles_degenerate_input() {
        let empty = TriMesh::default();
        let bvh = Bvh::build(&empty);
        assert!(
            bvh.raycast(&empty, Vec3::ZERO, Vec3::X, f32::INFINITY)
                .is_none()
        );
        assert!(
            bvh.closest_point(&empty, Vec3::ZERO, f32::INFINITY)
                .is_none()
        );

        // Many triangles sharing one centroid force median splits.
        let mut mesh = TriMesh::default();
        for i in 0..5000u32 {
            let s = 1.0 + i as f32 * 1e-3;
            mesh.positions.extend([
                Vec3::new(-s, -s, 0.0),
                Vec3::new(s, -s, 0.0),
                Vec3::new(0.0, 2.0 * s, 0.0),
            ]);
            mesh.indices.extend([3 * i, 3 * i + 1, 3 * i + 2]);
        }
        let bvh = Bvh::build(&mesh);
        bvh.validate(&mesh);
        let hit = bvh
            .raycast(&mesh, Vec3::new(0.0, 0.0, 5.0), Vec3::NEG_Z, f32::INFINITY)
            .unwrap();
        assert!((hit.t - 5.0).abs() < 1e-5);
    }

    #[test]
    fn respects_query_limits() {
        let mesh = torus(1.0, 0.25, 64, 32);
        let bvh = Bvh::build(&mesh);
        let origin = Vec3::new(0.0, 0.0, 5.0);
        assert!(
            bvh.raycast(&mesh, origin, Vec3::NEG_Z, 3.0).is_none(),
            "surface is beyond t_max"
        );
        assert!(bvh.raycast(&mesh, origin, Vec3::NEG_Z, 10.0).is_some());
        assert!(
            bvh.raycast(&mesh, origin, Vec3::Z, 10.0).is_none(),
            "pointing away"
        );
        assert!(
            bvh.closest_point(&mesh, Vec3::new(0.0, 3.0, 0.0), 1.0)
                .is_none()
        );
        let near = bvh
            .closest_point(&mesh, Vec3::new(1.0, 0.5, 0.0), 1.0)
            .unwrap();
        assert!((near.distance - 0.25).abs() < 1e-2);
    }
}
