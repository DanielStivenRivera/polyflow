//! Rasterization of the low-poly's UV layout: which triangle covers each
//! texel of the map, and where in that triangle the texel lies.
//!
//! Works like a GPU rasterizer: UVs are snapped to a fixed-point grid of
//! 1/256 texel and edge functions are evaluated exactly in integers, with the
//! top-left fill rule. Triangles sharing an edge therefore never both cover,
//! nor both miss, a sample on that edge, whatever their orientation.
//!
//! Texel `(x, y)` is row `y` from the top of the image; UV `v = 0` is the
//! bottom (the OBJ convention). Only the 0..1 UV tile is rasterized.
//! Where UV shells overlap, the triangle with the lowest index wins and the
//! extra coverage is counted.

use polyflow_core::glam::Vec2;
use rayon::prelude::*;

use crate::mesh::{BakeMesh, Corner};

/// Sub-texel precision of the fixed-point grid.
pub const SUBPIXEL_BITS: u32 = 8;
const ONE: i64 = 1 << SUBPIXEL_BITS;
/// Largest supported map side, so edge functions fit in `i64`.
pub const MAX_SIZE: u32 = 16384;
/// UVs further than this from the 0..1 tile are refused (same reason).
const MAX_UV: f32 = 128.0;
/// Marks texels no triangle covers.
pub const NO_TRIANGLE: u32 = u32::MAX;
/// Rows rasterized together by one thread.
const BAND: usize = 16;

/// For each texel: the covering triangle and the barycentric coordinates of
/// the sample in it (weights of corners 1 and 2, as [`BakeMesh::sample`]
/// takes them).
#[derive(Debug, Clone)]
pub struct UvRaster {
    pub width: u32,
    pub height: u32,
    /// Triangle per texel, row by row; [`NO_TRIANGLE`] where none.
    pub triangles: Vec<u32>,
    pub barycentrics: Vec<[f32; 2]>,
    /// Texels covered by some triangle.
    pub covered: usize,
    /// Extra coverages where UV shells overlap (each dropped one counts).
    pub overlaps: usize,
    /// Triangles with zero UV area, which cover nothing.
    pub degenerate: usize,
    /// Triangles refused because their UVs are absurdly far out.
    pub out_of_range: usize,
}

impl UvRaster {
    /// The triangle and barycentric coordinates at texel `(x, y)`.
    pub fn texel(&self, x: u32, y: u32) -> Option<(u32, [f32; 2])> {
        let i = y as usize * self.width as usize + x as usize;
        let t = self.triangles[i];
        (t != NO_TRIANGLE).then(|| (t, self.barycentrics[i]))
    }

    /// Fraction of the map covered.
    pub fn coverage(&self) -> f64 {
        self.covered as f64 / (self.width as f64 * self.height as f64).max(1.0)
    }
}

/// A triangle in fixed-point texel coordinates, oriented so its interior is
/// on the positive side of each edge.
#[derive(Debug, Clone, Copy)]
struct Setup {
    p: [[i64; 2]; 3],
    area: i64,
    /// Corners 1 and 2 were swapped to fix the orientation.
    swapped: bool,
    rows: [i64; 2],
    cols: [i64; 2],
}

/// Rasterizes `mesh`'s UVs at `width` x `height`, sampling each texel at
/// `offset` (in texels from its top-left corner; 0.5 is the center).
pub fn rasterize(mesh: &BakeMesh, width: u32, height: u32, offset: Vec2) -> UvRaster {
    assert!(
        (1..=MAX_SIZE).contains(&width) && (1..=MAX_SIZE).contains(&height),
        "map size {width}x{height} out of range"
    );
    let (w, h) = (width as usize, height as usize);
    let offset = [
        (offset.x * ONE as f32).round() as i64,
        (offset.y * ONE as f32).round() as i64,
    ];
    let mut out = UvRaster {
        width,
        height,
        triangles: vec![NO_TRIANGLE; w * h],
        barycentrics: vec![[0.0; 2]; w * h],
        covered: 0,
        overlaps: 0,
        degenerate: 0,
        out_of_range: 0,
    };

    // Set up every triangle and bin it into the bands of rows it touches.
    let mut setups: Vec<Option<Setup>> = Vec::with_capacity(mesh.triangle_count());
    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); h.div_ceil(BAND)];
    for t in 0..mesh.triangle_count() {
        let corners = mesh.triangle(t);
        // Also catches NaN.
        let in_range = |c: &&Corner| c.uv.is_finite() && c.uv.abs().max_element() <= MAX_UV;
        if !corners.iter().all(|c| in_range(&c)) {
            out.out_of_range += 1;
            setups.push(None);
            continue;
        }
        let fixed = |uv: Vec2| {
            [
                (uv.x as f64 * width as f64 * ONE as f64).round() as i64,
                ((1.0 - uv.y as f64) * height as f64 * ONE as f64).round() as i64,
            ]
        };
        let mut p = corners.map(|c| fixed(c.uv));
        let mut area = edge(p[0], p[1], p[2]);
        if area == 0 {
            out.degenerate += 1;
            setups.push(None);
            continue;
        }
        let swapped = area < 0;
        if swapped {
            p.swap(1, 2);
            area = -area;
        }
        // Texels whose sample point lies within the bounding box.
        let range = |axis: usize, size: usize| {
            let lo = p.iter().map(|q| q[axis]).min().unwrap() - offset[axis];
            let hi = p.iter().map(|q| q[axis]).max().unwrap() - offset[axis];
            [
                lo.div_euclid(ONE) + (lo.rem_euclid(ONE) != 0) as i64,
                hi.div_euclid(ONE),
            ]
            .map(|v| v.clamp(-1, size as i64))
        };
        let cols = range(0, w);
        let rows = range(1, h);
        let first = rows[0].max(0);
        let last = rows[1].min(h as i64 - 1);
        if cols[0] > cols[1].min(w as i64 - 1) || cols[1] < 0 || first > last {
            setups.push(None);
            continue;
        }
        setups.push(Some(Setup {
            p,
            area,
            swapped,
            rows: [first, last],
            cols: [cols[0].max(0), cols[1].min(w as i64 - 1)],
        }));
        for bin in &mut bins[first as usize / BAND..=last as usize / BAND] {
            bin.push(t as u32);
        }
    }

    let counts: Vec<(usize, usize)> = out
        .triangles
        .par_chunks_mut(BAND * w)
        .zip(out.barycentrics.par_chunks_mut(BAND * w))
        .zip(&bins)
        .enumerate()
        .map(|(band, ((triangles, barycentrics), bin))| {
            let row0 = (band * BAND) as i64;
            let rows = (triangles.len() / w) as i64;
            let (mut covered, mut overlaps) = (0, 0);
            for &t in bin {
                let s = setups[t as usize].expect("binned triangles are set up");
                let [a, b, c] = s.p;
                for y in s.rows[0].max(row0)..=s.rows[1].min(row0 + rows - 1) {
                    let py = y * ONE + offset[1];
                    for x in s.cols[0]..=s.cols[1] {
                        let q = [x * ONE + offset[0], py];
                        // Weights of a, b and c, times the area.
                        let wa = edge(b, c, q);
                        let wb = edge(c, a, q);
                        let wc = edge(a, b, q);
                        if !(inside(wa, b, c) && inside(wb, c, a) && inside(wc, a, b)) {
                            continue;
                        }
                        let i = (y - row0) as usize * w + x as usize;
                        if triangles[i] != NO_TRIANGLE {
                            overlaps += 1;
                            continue;
                        }
                        let area = s.area as f64;
                        let (wb, wc) = (wb as f64 / area, wc as f64 / area);
                        triangles[i] = t;
                        barycentrics[i] = if s.swapped {
                            [wc as f32, wb as f32]
                        } else {
                            [wb as f32, wc as f32]
                        };
                        covered += 1;
                    }
                }
            }
            (covered, overlaps)
        })
        .collect();
    for (covered, overlaps) in counts {
        out.covered += covered;
        out.overlaps += overlaps;
    }
    out
}

/// Twice the signed area of `(a, b, q)`: positive when `q` is on the
/// interior side of the edge `a -> b` of a positively oriented triangle.
#[inline]
fn edge(a: [i64; 2], b: [i64; 2], q: [i64; 2]) -> i64 {
    (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
}

/// Top-left rule: a sample exactly on an edge belongs to the triangle for
/// which that edge is a left or top one, i.e. runs up the image (or right,
/// if horizontal) in the positive orientation. A shared edge runs in opposite
/// directions in its two triangles, so exactly one owns it.
#[inline]
fn inside(weight: i64, from: [i64; 2], to: [i64; 2]) -> bool {
    let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
    weight > 0 || (weight == 0 && (dy < 0 || (dy == 0 && dx > 0)))
}

#[cfg(test)]
mod tests {
    use polyflow_core::glam::Vec3;
    use polyflow_core::halfedge::HalfEdgeMesh;

    use super::*;
    use crate::mesh::tests::uv_grid;

    /// Small deterministic PRNG.
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n
        }
    }

    /// A UV square split into an `n` x `n` grid of triangle pairs with random
    /// diagonals and random corner orders, its vertices jittered on the
    /// fixed-point grid, possibly onto sample points.
    fn jittered_square(n: usize, rng: &mut Rng) -> HalfEdgeMesh {
        let mut mesh = HalfEdgeMesh::new();
        let mut ids = Vec::new();
        let mut uvs = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let inner = |k: usize| k > 0 && k < n;
                let jitter = |k: usize, rng: &mut Rng| {
                    if inner(k) {
                        (rng.below(9) as f32 - 4.0) / 64.0
                    } else {
                        0.0
                    }
                };
                let uv = Vec2::new(
                    (i as f32 + jitter(i, rng)) / n as f32,
                    (j as f32 + jitter(j, rng)) / n as f32,
                );
                ids.push(mesh.add_vertex(uv.extend(0.0)));
                uvs.push(uv);
            }
        }
        for j in 0..n {
            for i in 0..n {
                let k = j * (n + 1) + i;
                let quad = [k, k + 1, k + n + 2, k + n + 1];
                let tris = if rng.below(2) == 0 {
                    [[0, 1, 2], [0, 2, 3]]
                } else {
                    [[0, 1, 3], [1, 2, 3]]
                };
                for tri in tris {
                    let corners = tri.map(|c| quad[c]);
                    let f = mesh.add_face(&corners.map(|c| ids[c])).unwrap();
                    mesh.set_face_uvs(f, &corners.map(|c| uvs[c]));
                }
            }
        }
        mesh
    }

    #[test]
    fn a_tiled_square_covers_every_texel_exactly_once() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for case in 0..40 {
            let n = 1 + rng.below(6) as usize;
            let size = [8, 16, 32, 37][case % 4];
            let mesh = BakeMesh::from_retopo(&jittered_square(n, &mut rng));
            // Sample points on texel corners hit vertices and edges exactly.
            for offset in [Vec2::ZERO, Vec2::splat(0.5), Vec2::new(0.25, 0.75)] {
                let raster = rasterize(&mesh, size, size, offset);
                assert_eq!(raster.overlaps, 0, "case {case}, {offset}");
                // At offset 0 the left and top borders are sampled: they are
                // top-left edges, so they count.
                assert_eq!(
                    raster.covered,
                    (size * size) as usize,
                    "case {case}, {offset}"
                );
            }
        }
    }

    #[test]
    fn barycentrics_reproduce_the_uv_mapping() {
        // Positions equal UVs scaled by 3, so interpolated positions tell
        // where each sample landed.
        let mesh = BakeMesh::from_retopo(&uv_grid(3, false));
        let raster = rasterize(&mesh, 64, 32, Vec2::splat(0.5));
        assert_eq!(raster.covered, 64 * 32);
        for y in 0..32 {
            for x in 0..64 {
                let (t, bary) = raster.texel(x, y).unwrap();
                let p = mesh.sample(t as usize, bary).position;
                let uv = Vec2::new((x as f32 + 0.5) / 64.0, 1.0 - (y as f32 + 0.5) / 32.0);
                // UVs snap to 1/256 texel (3 / (64 * 256) in position).
                assert!(p.truncate().distance(uv * 3.0) < 2e-4, "({x}, {y}): {p}");
            }
        }
        // Mirrored UVs (swapped corners internally) interpolate just as well.
        let mesh = BakeMesh::from_retopo(&uv_grid(3, true));
        let raster = rasterize(&mesh, 16, 16, Vec2::splat(0.5));
        let (t, bary) = raster.texel(0, 15).unwrap();
        let p = mesh.sample(t as usize, bary).position;
        let expected = Vec3::new((1.0 - 0.5 / 16.0) * 3.0, 0.5 / 16.0 * 3.0, 0.0);
        assert!(p.distance(expected) < 8e-4, "{p}");
    }

    #[test]
    fn overlaps_degenerates_and_far_uvs_are_reported() {
        let mut mesh = HalfEdgeMesh::new();
        let v = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ONE].map(|p| mesh.add_vertex(p));
        let uv = [Vec2::ZERO, Vec2::X, Vec2::Y];
        // Two triangles with identical UVs, one with a collapsed UV, one far away.
        for (corners, uvs) in [
            ([v[0], v[1], v[2]], uv),
            ([v[1], v[3], v[2]], uv),
            ([v[0], v[2], v[3]], [Vec2::ZERO, Vec2::ZERO, Vec2::X]),
        ] {
            let f = mesh.add_face(&corners).unwrap();
            mesh.set_face_uvs(f, &uvs);
        }
        let mut bake = BakeMesh::from_retopo(&mesh);
        bake.corners.extend_from_within(0..3);
        bake.corners[9].uv = Vec2::new(1e9, 0.0);
        bake.source_faces.push(bake.source_faces[0]);
        let raster = rasterize(&bake, 10, 10, Vec2::splat(0.5));
        // Below the diagonal: 45 texels; the 10 centered on the diagonal
        // itself go to whatever lies on its other side (nothing here).
        assert_eq!(raster.covered, 45);
        assert_eq!(raster.overlaps, 45);
        assert_eq!(raster.degenerate, 1);
        assert_eq!(raster.out_of_range, 1);
        assert!(raster.texel(9, 0).is_none());
        assert_eq!(raster.texel(0, 9).unwrap().0, 0, "the lowest index wins");
    }
}
