//! The CPU reference baker: tangent- and object-space normals and ambient
//! occlusion, projected from the reference mesh onto the low-poly's UVs.
//!
//! Projection: for each texel, a ray starts in front of the low-poly
//! surface and travels back through it, up to `rear_distance` behind; the
//! first reference surface it meets is the one baked. Without a cage the
//! ray starts `frontal_distance` out along the smooth normal; with one (see
//! [`crate::cage`]) it starts at the texel's point on the cage. Texels whose ray finds
//! nothing are "missed": they get neutral values (flat normal, no occlusion)
//! and are reported, as they usually mean the distances are too small.
//!
//! Results only depend on the inputs, not on the number of threads.

use std::f32::consts::TAU;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::mesh::TriMesh;
use polyflow_spatial::Bvh;
use rayon::prelude::*;

use crate::mesh::BakeMesh;
use crate::padding::Padding;
use crate::raster::rasterize;

/// The high-poly reference: its mesh and a BVH built from it.
#[derive(Debug, Clone, Copy)]
pub struct Reference<'a> {
    pub mesh: &'a TriMesh,
    pub bvh: &'a Bvh,
}

/// Which way green points in tangent-space normal maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormalY {
    /// Green up (+Y): Blender, Unity, Godot, Substance's default.
    #[default]
    OpenGl,
    /// Green down (-Y): Unreal, 3ds Max, CryEngine.
    DirectX,
}

/// Which maps to bake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapSet {
    pub tangent_normal: bool,
    pub object_normal: bool,
    pub ambient_occlusion: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BakeSettings {
    pub width: u32,
    pub height: u32,
    /// How far in front of the low-poly rays start, in scene units. Unused
    /// with a cage, which says where rays start.
    pub frontal_distance: f32,
    /// How far behind the low-poly rays reach, in scene units.
    pub rear_distance: f32,
    pub maps: MapSet,
    pub normal_y: NormalY,
    /// Rays per texel for ambient occlusion.
    pub ao_samples: u32,
    /// Occluders further than this do not count, in scene units.
    pub ao_distance: f32,
    /// Samples per texel along each axis (1 to 8): `samples`² per texel,
    /// antialiasing edges in the detail and along UV borders.
    pub samples: u32,
    /// How many texels islands are extended by beyond their UV borders, so
    /// filtering and mipmaps do not pull in the background; `None` fills the
    /// whole map.
    pub padding: Option<u32>,
}

/// Largest supported [`BakeSettings::samples`].
pub const MAX_SAMPLES: u32 = 8;

impl BakeSettings {
    /// What [`BakeControl::rows_done`] reaches when the bake is done: every
    /// row once per sample.
    pub fn progress_rows(&self) -> u32 {
        let n = self.samples.clamp(1, MAX_SAMPLES);
        self.height * n * n
    }

    /// Settings scaled to a reference mesh of bounding-sphere `radius`:
    /// rays reach 3% of it in front and behind, occlusion 25%.
    pub fn for_scene(width: u32, height: u32, radius: f32) -> Self {
        Self {
            width,
            height,
            frontal_distance: radius * 0.03,
            rear_distance: radius * 0.03,
            maps: MapSet {
                tangent_normal: true,
                object_normal: true,
                ambient_occlusion: true,
            },
            normal_y: NormalY::OpenGl,
            ao_samples: 32,
            ao_distance: radius * 0.25,
            samples: 1,
            padding: Some(16),
        }
    }
}

/// A baked image: `channels` floats per texel, row by row from the top.
/// Normals are stored as colors (`n * 0.5 + 0.5`), occlusion as visibility
/// (1 = unoccluded).
#[derive(Debug, Clone, PartialEq)]
pub struct Map {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub data: Vec<f32>,
}

impl Map {
    fn filled(width: u32, height: u32, value: &[f32]) -> Self {
        Self {
            width,
            height,
            channels: value.len(),
            data: value.repeat(width as usize * height as usize),
        }
    }

    pub fn texel(&self, x: u32, y: u32) -> &[f32] {
        let i = (y as usize * self.width as usize + x as usize) * self.channels;
        &self.data[i..i + self.channels]
    }
}

/// What happened at a texel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum Coverage {
    /// Outside the UV layout.
    #[default]
    Empty,
    /// Inside the layout, but the ray found no reference surface.
    Missed,
    Hit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Baked {
    pub tangent_normal: Option<Map>,
    pub object_normal: Option<Map>,
    pub ambient_occlusion: Option<Map>,
    /// Per texel, row by row; the mask for edge padding.
    pub coverage: Vec<Coverage>,
    pub hits: usize,
    pub misses: usize,
}

/// Progress and cancellation, shared with the thread that started the bake.
#[derive(Debug, Default)]
pub struct BakeControl {
    cancel: AtomicBool,
    rows_done: AtomicU32,
}

impl BakeControl {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Rows finished so far, out of [`BakeSettings::progress_rows`].
    pub fn rows_done(&self) -> u32 {
        self.rows_done.load(Ordering::Relaxed)
    }

    /// Whether [`Self::cancel`] was called; bakers check it between pieces
    /// of work.
    pub fn cancel_requested(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Records `rows` more rows done; for bakers.
    pub fn add_rows(&self, rows: u32) {
        self.rows_done.fetch_add(rows, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BakeError {
    #[error("the bake was cancelled")]
    Cancelled,
}

/// Neutral values for texels with nothing baked.
const FLAT_NORMAL: [f32; 3] = [0.5, 0.5, 1.0];
const ZERO_NORMAL: [f32; 3] = [0.5, 0.5, 0.5];
const UNOCCLUDED: [f32; 1] = [1.0];

/// One row of every output, accumulated over a texel's samples.
struct Row<'a> {
    y: u32,
    /// Sum of tangent-space normals (unit vectors), before encoding.
    tangent: Option<&'a mut [f32]>,
    /// Sum of reference normals; of low-poly normals while nothing was hit.
    object: Option<&'a mut [f32]>,
    /// Sum of visibilities.
    ao: Option<&'a mut [f32]>,
    /// Samples that hit the reference.
    hits: &'a mut [u16],
    /// Samples inside the UV layout.
    covered: &'a mut [u16],
}

/// Bakes `settings.maps` from `reference` onto `low`'s UVs.
///
/// Each texel is sampled `settings.samples`² times on a regular grid; maps
/// average the samples that hit the reference (normals are averaged as
/// vectors and renormalized). Then edge padding extends the islands.
pub fn bake(
    low: &BakeMesh,
    reference: Reference<'_>,
    settings: &BakeSettings,
    control: &BakeControl,
) -> Result<Baked, BakeError> {
    let (width, height) = (settings.width, settings.height);
    let BakeSums {
        tangent_normal: mut tangent,
        object_normal: mut object,
        ambient_occlusion: mut occlusion,
        mut hits,
        mut covered,
    } = BakeSums::new(settings);
    let ao = AoSampler::new(settings, reference.mesh);

    let n = settings.samples.clamp(1, MAX_SAMPLES);
    let w = width as usize;
    for pass in 0..n * n {
        let offset = Vec2::new(
            ((pass % n) as f32 + 0.5) / n as f32,
            ((pass / n) as f32 + 0.5) / n as f32,
        );
        let raster = rasterize(low, width, height, offset);
        let mut tangent_rows = row_slices(&mut tangent, height);
        let mut object_rows = row_slices(&mut object, height);
        let mut ao_rows = row_slices(&mut occlusion, height);
        let rows: Vec<Row<'_>> = hits
            .chunks_mut(w)
            .zip(covered.chunks_mut(w))
            .enumerate()
            .map(|(y, (hits, covered))| Row {
                y: y as u32,
                tangent: tangent_rows.next().flatten(),
                object: object_rows.next().flatten(),
                ao: ao_rows.next().flatten(),
                hits,
                covered,
            })
            .collect();
        rows.into_par_iter().for_each(|mut row| {
            if control.cancel.load(Ordering::Relaxed) {
                return;
            }
            bake_row(&mut row, low, reference, &raster, settings, &ao, pass);
            control.rows_done.fetch_add(1, Ordering::Relaxed);
        });
        if control.cancel.load(Ordering::Relaxed) {
            return Err(BakeError::Cancelled);
        }
    }

    let sums = BakeSums {
        tangent_normal: tangent,
        object_normal: object,
        ambient_occlusion: occlusion,
        hits,
        covered,
    };
    Ok(finish(sums, settings))
}

/// What a bake adds up over each texel's samples, before [`finish`] turns
/// it into maps. Any baker (this one, the GPU one) fills these.
#[derive(Debug, Clone, PartialEq)]
pub struct BakeSums {
    /// Sum over the samples that hit of the tangent-space normal (unit
    /// vector, green flipped for DirectX), not yet encoded as a color.
    pub tangent_normal: Option<Map>,
    /// Sum over the samples that hit of the reference normal; while no sample
    /// of the texel has hit, of the low-poly's normal at the missed samples.
    pub object_normal: Option<Map>,
    /// Sum over the samples that hit of the visibility.
    pub ambient_occlusion: Option<Map>,
    /// Samples that hit, per texel.
    pub hits: Vec<u16>,
    /// Samples inside the UV layout, per texel.
    pub covered: Vec<u16>,
}

impl BakeSums {
    /// All zeros, with the maps `settings` asks for.
    pub fn new(settings: &BakeSettings) -> Self {
        let (width, height) = (settings.width, settings.height);
        let texels = width as usize * height as usize;
        let zeros = |wanted: bool, channels: usize| {
            wanted.then(|| Map::filled(width, height, &vec![0.0; channels]))
        };
        let maps = settings.maps;
        Self {
            tangent_normal: zeros(maps.tangent_normal, 3),
            object_normal: zeros(maps.object_normal, 3),
            ambient_occlusion: zeros(maps.ambient_occlusion, 1),
            hits: vec![0; texels],
            covered: vec![0; texels],
        }
    }
}

/// Turns sums into maps: averages, normalizes and encodes them, puts neutral
/// values where nothing was hit, then pads the islands.
pub fn finish(sums: BakeSums, settings: &BakeSettings) -> Baked {
    let (width, height) = (settings.width, settings.height);
    let BakeSums {
        tangent_normal: tangent,
        object_normal: object,
        ambient_occlusion: occlusion,
        hits,
        covered,
    } = sums;
    // Turn sums into values; neutral values where nothing was hit.
    let mut out = Baked {
        tangent_normal: tangent,
        object_normal: object,
        ambient_occlusion: occlusion,
        coverage: hits
            .iter()
            .zip(&covered)
            .map(|(&h, &c)| match (h, c) {
                (_, 0) => Coverage::Empty,
                (0, _) => Coverage::Missed,
                _ => Coverage::Hit,
            })
            .collect(),
        hits: 0,
        misses: 0,
    };
    out.hits = out.coverage.iter().filter(|&&c| c == Coverage::Hit).count();
    out.misses = out
        .coverage
        .iter()
        .filter(|&&c| c == Coverage::Missed)
        .count();
    let coverage = &out.coverage;
    if let Some(map) = &mut out.tangent_normal {
        map.data
            .par_chunks_mut(3)
            .zip(coverage)
            .for_each(|(t, &c)| {
                let value = match c {
                    Coverage::Hit => color(Vec3::from_slice(t).normalize_or(Vec3::Z)),
                    _ => FLAT_NORMAL,
                };
                t.copy_from_slice(&value);
            });
    }
    if let Some(map) = &mut out.object_normal {
        map.data
            .par_chunks_mut(3)
            .zip(coverage)
            .for_each(|(t, &c)| {
                let value = match c {
                    Coverage::Empty => ZERO_NORMAL,
                    _ => color(Vec3::from_slice(t).normalize_or_zero()),
                };
                t.copy_from_slice(&value);
            });
    }
    if let Some(map) = &mut out.ambient_occlusion {
        map.data
            .par_iter_mut()
            .zip(&hits)
            .for_each(|(v, &h)| *v = if h > 0 { *v / h as f32 } else { UNOCCLUDED[0] });
    }

    let padding = Padding::new(width, height, &out.coverage, settings.padding);
    for map in [
        &mut out.tangent_normal,
        &mut out.object_normal,
        &mut out.ambient_occlusion,
    ]
    .into_iter()
    .flatten()
    {
        padding.apply(map);
    }
    out
}

/// A map's rows, or as many `None`s if the map is not baked.
fn row_slices(map: &mut Option<Map>, height: u32) -> impl Iterator<Item = Option<&mut [f32]>> {
    let rows: Vec<Option<&mut [f32]>> = match map {
        Some(m) => {
            let len = m.width as usize * m.channels;
            m.data.chunks_mut(len).map(Some).collect()
        }
        None => (0..height).map(|_| None).collect(),
    };
    rows.into_iter()
}

/// Adds sample `pass` of every texel of `row` to the sums.
fn bake_row(
    row: &mut Row<'_>,
    low: &BakeMesh,
    reference: Reference<'_>,
    raster: &crate::UvRaster,
    settings: &BakeSettings,
    ao: &AoSampler,
    pass: u32,
) {
    let y = row.y;
    for x in 0..settings.width {
        let Some((t, bary)) = raster.texel(x, y) else {
            continue;
        };
        let xi = x as usize;
        row.covered[xi] += 1;
        let sample = low.sample(t as usize, bary);
        let normal = sample.frame.normal.normalize_or_zero();
        let hit = ray(low, &sample, normal, settings).and_then(|(origin, dir, reach)| {
            reference.bvh.raycast(reference.mesh, origin, dir, reach)
        });
        let add = |slot: &mut [f32], v: Vec3| {
            for (s, v) in slot.iter_mut().zip(v.to_array()) {
                *s += v;
            }
        };
        let Some(hit) = hit else {
            // Until something is hit, the object map shows the low-poly.
            if row.hits[xi] == 0
                && let Some(object) = &mut row.object
            {
                add(&mut object[xi * 3..xi * 3 + 3], normal);
            }
            continue;
        };
        let first_hit = row.hits[xi] == 0;
        row.hits[xi] += 1;

        let surface = SurfacePoint::at(reference.mesh, hit.triangle, hit.u, hit.v);
        if let Some(tangent) = &mut row.tangent {
            let mut n = sample.frame.encode(surface.normal);
            if settings.normal_y == NormalY::DirectX {
                n.y = -n.y;
            }
            add(&mut tangent[xi * 3..xi * 3 + 3], n);
        }
        if let Some(object) = &mut row.object {
            let slot = &mut object[xi * 3..xi * 3 + 3];
            if first_hit {
                // Drop the low-poly normals summed by earlier misses.
                slot.fill(0.0);
            }
            add(slot, surface.normal);
        }
        if let Some(occlusion) = &mut row.ao {
            occlusion[xi] += ao.visibility(reference, &surface, x, y, pass);
        }
    }
}

/// The texel's ray: origin, unit direction and length. From the cage point
/// through the low-poly point if there is a cage, else along the normal.
fn ray(
    low: &BakeMesh,
    sample: &crate::SurfaceSample,
    normal: Vec3,
    settings: &BakeSettings,
) -> Option<(Vec3, Vec3, f32)> {
    if low.has_cage {
        let to_surface = sample.position - sample.cage;
        let length = to_surface.length();
        // Where the cage touches the low-poly, start on the surface.
        if length > 1e-6 * (sample.position.abs().max_element() + 1.0) {
            return Some((
                sample.cage,
                to_surface / length,
                length + settings.rear_distance,
            ));
        }
        return (normal != Vec3::ZERO).then_some((
            sample.position,
            -normal,
            settings.rear_distance,
        ));
    }
    (normal != Vec3::ZERO).then(|| {
        let origin = sample.position + normal * settings.frontal_distance;
        (
            origin,
            -normal,
            settings.frontal_distance + settings.rear_distance,
        )
    })
}

/// A unit vector as a color.
fn color(n: Vec3) -> [f32; 3] {
    (n * 0.5 + 0.5).to_array()
}

/// A point on the reference surface.
struct SurfacePoint {
    position: Vec3,
    /// Interpolated vertex normal, unit length.
    normal: Vec3,
    /// The triangle's own normal, on the same side as `normal`.
    face_normal: Vec3,
}

impl SurfacePoint {
    fn at(mesh: &TriMesh, triangle: u32, u: f32, v: f32) -> Self {
        let i = triangle as usize * 3;
        let [a, b, c] = [0, 1, 2].map(|k| mesh.indices[i + k] as usize);
        let w = [1.0 - u - v, u, v];
        let position =
            mesh.positions[a] * w[0] + mesh.positions[b] * w[1] + mesh.positions[c] * w[2];
        let face = (mesh.positions[b] - mesh.positions[a])
            .cross(mesh.positions[c] - mesh.positions[a])
            .normalize_or_zero();
        let smooth = mesh.normals[a] * w[0] + mesh.normals[b] * w[1] + mesh.normals[c] * w[2];
        let normal = smooth.normalize_or(face);
        let face_normal = if face.dot(normal) < 0.0 { -face } else { face };
        Self {
            position,
            normal,
            face_normal: face_normal.normalize_or(normal),
        }
    }
}

/// Cosine-weighted hemisphere sampling for ambient occlusion.
struct AoSampler {
    /// The same low-discrepancy (Hammersley) points for every texel, each
    /// texel shifting them by its own random offset.
    points: Vec<Vec2>,
    distance: f32,
    /// How far rays start off the surface, so they do not hit it.
    bias: f32,
}

/// The sample points every texel's occlusion rays use, before each texel
/// shifts them (see [`texel_random`]).
pub fn ao_points(settings: &BakeSettings) -> Vec<Vec2> {
    let n = settings.ao_samples.max(1);
    (0..n)
        .map(|i| Vec2::new((i as f32 + 0.5) / n as f32, radical_inverse(i)))
        .collect()
}

/// How far occlusion rays start off the reference surface.
pub fn ao_bias(reference: &TriMesh) -> f32 {
    reference.bounds().radius() * 1e-5
}

impl AoSampler {
    fn new(settings: &BakeSettings, mesh: &TriMesh) -> Self {
        Self {
            points: ao_points(settings),
            distance: settings.ao_distance,
            bias: ao_bias(mesh),
        }
    }

    /// Fraction of the hemisphere above `surface` that sees nothing within
    /// `distance`, cosine-weighted.
    fn visibility(
        &self,
        reference: Reference<'_>,
        surface: &SurfacePoint,
        x: u32,
        y: u32,
        pass: u32,
    ) -> f32 {
        let n = surface.normal;
        let (t, b) = n.any_orthonormal_pair();
        let origin = surface.position + surface.face_normal * self.bias;
        let shift = texel_random(x, y, pass);
        let mut occluded = 0;
        for &p in &self.points {
            let p = (p + shift).fract();
            let r = p.x.sqrt();
            let (sin, cos) = (TAU * p.y).sin_cos();
            let mut dir = t * (r * cos) + b * (r * sin) + n * (1.0 - p.x).max(0.0).sqrt();
            // Around smooth normals some directions dip under the actual
            // triangle; reflect them back above it.
            let below = dir.dot(surface.face_normal);
            if below < 0.0 {
                dir -= surface.face_normal * (2.0 * below);
            }
            if reference
                .bvh
                .occluded(reference.mesh, origin, dir, self.distance)
            {
                occluded += 1;
            }
        }
        1.0 - occluded as f32 / self.points.len() as f32
    }
}

/// Van der Corput sequence in base 2.
fn radical_inverse(i: u32) -> f32 {
    i.reverse_bits() as f32 * (1.0 / 4_294_967_296.0)
}

/// Two well-mixed numbers in 0..1 for a texel's sample (PCG-style hash),
/// shifting its occlusion rays.
pub fn texel_random(x: u32, y: u32, pass: u32) -> Vec2 {
    let hash = |mut v: u32| {
        v = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
        let w = ((v >> ((v >> 28) + 4)) ^ v).wrapping_mul(277_803_737);
        (w >> 22) ^ w
    };
    let a = hash(x ^ hash(y ^ hash(pass)));
    let b = hash(a);
    Vec2::new(a as f32, b as f32) * (1.0 / 4_294_967_296.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::tests::uv_grid;

    /// An `n` x `n` grid of quads over `[min, max]²` with heights `z(x, y)`,
    /// facing +Z (or -Z if `down`).
    fn sheet(min: f32, max: f32, n: usize, down: bool, z: impl Fn(f32, f32) -> f32) -> TriMesh {
        sheet_rect([min, max], [min, max], n, down, z)
    }

    /// Like [`sheet`] over `xs` x `ys`.
    fn sheet_rect(
        xs: [f32; 2],
        ys: [f32; 2],
        n: usize,
        down: bool,
        z: impl Fn(f32, f32) -> f32,
    ) -> TriMesh {
        let mut mesh = TriMesh::default();
        let step = [(xs[1] - xs[0]) / n as f32, (ys[1] - ys[0]) / n as f32];
        for j in 0..=n {
            for i in 0..=n {
                let (x, y) = (xs[0] + i as f32 * step[0], ys[0] + j as f32 * step[1]);
                mesh.positions.push(Vec3::new(x, y, z(x, y)));
            }
        }
        for j in 0..n {
            for i in 0..n {
                let k = (j * (n + 1) + i) as u32;
                let row = n as u32 + 1;
                let mut tris = [[k, k + 1, k + row + 1], [k, k + row + 1, k + row]];
                if down {
                    tris.iter_mut().for_each(|t| t.swap(1, 2));
                }
                mesh.indices.extend(tris.iter().flatten());
            }
        }
        mesh.compute_normals();
        mesh
    }

    fn merge(meshes: &[TriMesh]) -> TriMesh {
        let mut out = TriMesh::default();
        for m in meshes {
            let base = out.positions.len() as u32;
            out.positions.extend(&m.positions);
            out.normals.extend(&m.normals);
            out.indices.extend(m.indices.iter().map(|i| i + base));
        }
        out
    }

    fn settings(size: u32, frontal: f32, rear: f32) -> BakeSettings {
        BakeSettings {
            frontal_distance: frontal,
            rear_distance: rear,
            ..BakeSettings::for_scene(size, size, 1.0)
        }
    }

    fn run(low: &BakeMesh, high: &TriMesh, settings: &BakeSettings) -> Baked {
        let bvh = Bvh::build(high);
        let reference = Reference {
            mesh: high,
            bvh: &bvh,
        };
        bake(low, reference, settings, &BakeControl::default()).unwrap()
    }

    /// Every hit texel of `map` equals `expected`.
    fn assert_uniform(map: &Map, coverage: &[Coverage], expected: [f32; 3], what: &str) {
        for (i, texel) in map.data.chunks(3).enumerate() {
            if coverage[i] == Coverage::Hit {
                let err = Vec3::from_slice(texel).distance(Vec3::from_array(expected));
                assert!(
                    err < 1e-4,
                    "{what}: texel {i} is {texel:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn a_tilted_plane_bakes_its_slope() {
        // The low-poly is the 4x4 plane at z = 0; the reference a plane
        // tilted by 20° around Y, then around X.
        let k = 20f32.to_radians().tan();
        for (around_y, mirrored, normal_y) in [
            (true, false, NormalY::OpenGl),
            (true, true, NormalY::OpenGl),
            (false, false, NormalY::OpenGl),
            (false, false, NormalY::DirectX),
        ] {
            let low = BakeMesh::from_retopo(&uv_grid(4, mirrored));
            let high = sheet(-1.0, 5.0, 12, false, |x, y| {
                0.3 + k * if around_y { x - 2.0 } else { y - 2.0 }
            });
            let mut settings = settings(32, 2.0, 2.0);
            settings.normal_y = normal_y;
            let baked = run(&low, &high, &settings);
            assert_eq!((baked.hits, baked.misses), (32 * 32, 0));

            let n = if around_y {
                Vec3::new(-k, 0.0, 1.0)
            } else {
                Vec3::new(0.0, -k, 1.0)
            }
            .normalize();
            assert_uniform(
                baked.object_normal.as_ref().unwrap(),
                &baked.coverage,
                color(n),
                "object",
            );
            // The tangent frame is (+X, +Y, +Z), or (-X, +Y, +Z) when mirrored;
            // DirectX maps store -Y.
            let mut tangent = n;
            if mirrored {
                tangent.x = -tangent.x;
            }
            if normal_y == NormalY::DirectX {
                tangent.y = -tangent.y;
            }
            let what = format!("tangent ({around_y}, {mirrored}, {normal_y:?})");
            assert_uniform(
                baked.tangent_normal.as_ref().unwrap(),
                &baked.coverage,
                color(tangent),
                &what,
            );
        }
    }

    #[test]
    fn rays_start_from_the_cage() {
        use crate::cage::Cage;
        let retopo = uv_grid(4, false);
        // The reference: a sheet 0.5 below the low-poly, only up to x = 1.75.
        let high = sheet(-1.0, 1.75, 11, false, |_, _| -0.5);
        let settings = settings(32, 0.1, 1.0);
        // 8 texels per unit along x; count the hit columns of row 28, at
        // y = (1 - 28.5 / 32) * 4 = 0.44, within the sheet.
        let hit_columns = |baked: &Baked| {
            (0..32)
                .filter(|&x| baked.coverage[28 * 32 + x] == Coverage::Hit)
                .count()
        };

        // Straight down: texel centers with x < 1.75 hit, (i + 0.5) / 8 < 1.75.
        let plain = run(&BakeMesh::from_retopo(&retopo), &high, &settings);
        assert_eq!(hit_columns(&plain), 14);
        // An automatic cage on a flat low-poly casts the very same rays.
        let auto = BakeMesh::with_cage(&retopo, Some(&Cage::automatic(&retopo, 0.1)));
        assert_eq!(run(&auto, &high, &settings), plain);

        // A cage shifted by (0.5, 0, 1): rays slant, crossing z = -0.5 at
        // x - 0.25, so texels up to x < 2 hit.
        let shifted: Vec<Vec3> = retopo.vertex_slots().flatten().collect();
        let mut moved = retopo.clone();
        let ids: Vec<_> = moved.vertices().collect();
        for (v, p) in ids.into_iter().zip(shifted) {
            moved.set_position(v, p + Vec3::new(0.5, 0.0, 1.0));
        }
        let cage = Cage::from_mesh(&retopo, &moved).unwrap();
        let slanted = run(&BakeMesh::with_cage(&retopo, Some(&cage)), &high, &settings);
        assert_eq!(hit_columns(&slanted), 16);
        assert_eq!(slanted.misses + slanted.hits, 32 * 32);
    }

    #[test]
    fn samples_average_within_a_texel() {
        // The reference steps at x = 2.06: flat to the left, tilted by 30° to
        // the right. Texel column 16 spans x 2.0..2.125; its center (2.0625)
        // is right of the step, but of its 2x2 samples (x 2.031 and 2.094)
        // half fall on each side.
        let low = BakeMesh::from_retopo(&uv_grid(4, false));
        let k = 30f32.to_radians().tan();
        let high = merge(&[
            sheet_rect([-1.0, 2.06], [-1.0, 5.0], 8, false, |_, _| 0.2),
            sheet_rect([2.06, 5.0], [-1.0, 5.0], 8, false, |x, _| {
                0.1 + k * (x - 2.06)
            }),
        ]);
        let tilted = Vec3::new(-k, 0.0, 1.0).normalize();
        let texel = |baked: &Baked, x: u32| {
            Vec3::from_slice(baked.tangent_normal.as_ref().unwrap().texel(x, 10))
        };
        let mut settings = settings(32, 2.0, 2.0);
        let single = run(&low, &high, &settings);
        assert!(texel(&single, 16).distance(Vec3::from(color(tilted))) < 1e-4);
        assert!(texel(&single, 15).distance(Vec3::from(FLAT_NORMAL)) < 1e-4);

        settings.samples = 2;
        assert_eq!(settings.progress_rows(), 32 * 4);
        let control = BakeControl::default();
        let bvh = Bvh::build(&high);
        let reference = Reference {
            mesh: &high,
            bvh: &bvh,
        };
        let multi = bake(&low, reference, &settings, &control).unwrap();
        assert_eq!(control.rows_done(), settings.progress_rows());
        let mixed = color((Vec3::Z + tilted).normalize());
        assert!(
            texel(&multi, 16).distance(Vec3::from(mixed)) < 1e-4,
            "{}",
            texel(&multi, 16)
        );
        assert!(texel(&multi, 15).distance(Vec3::from(FLAT_NORMAL)) < 1e-4);
        assert!(texel(&multi, 17).distance(Vec3::from(color(tilted))) < 1e-4);
        assert_eq!((multi.hits, multi.misses), (32 * 32, 0));
    }

    #[test]
    fn padding_fills_the_background() {
        // The UV layout covers only the lower-left quarter of the map.
        let mut retopo = uv_grid(2, false);
        let faces: Vec<_> = retopo.faces().collect();
        for f in faces {
            let uvs: Vec<_> = retopo
                .face_uvs(f)
                .unwrap()
                .iter()
                .map(|uv| *uv * 0.5)
                .collect();
            retopo.set_face_uvs(f, &uvs);
        }
        let low = BakeMesh::from_retopo(&retopo);
        let high = sheet(-1.0, 3.0, 4, false, |x, _| 0.3 * x);
        let is_background = |t: &[f32]| t == ZERO_NORMAL;
        let mut settings = settings(16, 2.0, 2.0);
        settings.padding = Some(0);
        let bare = run(&low, &high, &settings);
        assert_eq!(bare.hits, 64);
        let object = bare.object_normal.unwrap();
        assert_eq!(
            object.data.chunks(3).filter(|t| is_background(t)).count(),
            256 - 64
        );

        settings.padding = Some(2);
        let padded = run(&low, &high, &settings).object_normal.unwrap();
        // The 8x8 island grows by 2 texels up and to the right.
        assert_eq!(
            padded.data.chunks(3).filter(|t| is_background(t)).count(),
            256 - 100
        );
        settings.padding = None;
        let full = run(&low, &high, &settings).object_normal.unwrap();
        assert!(full.data.chunks(3).all(|t| !is_background(t)));
        assert_eq!(
            full.texel(15, 0),
            object.texel(7, 8),
            "copied from the island's corner"
        );
    }

    #[test]
    fn rays_that_fall_short_are_misses() {
        let low = BakeMesh::from_retopo(&uv_grid(2, false));
        let high = sheet(-1.0, 3.0, 4, false, |_, _| 1.0);
        let baked = run(&low, &high, &settings(16, 0.5, 0.5));
        assert_eq!((baked.hits, baked.misses), (0, 16 * 16));
        assert!(baked.coverage.iter().all(|&c| c == Coverage::Missed));
        let tangent = baked.tangent_normal.unwrap();
        assert!(tangent.data.chunks(3).all(|t| t == FLAT_NORMAL));
        let object = baked.object_normal.unwrap();
        assert_eq!(object.texel(3, 3), color(Vec3::Z));
        assert!(
            baked
                .ambient_occlusion
                .unwrap()
                .data
                .iter()
                .all(|&v| v == 1.0)
        );
        // Reaching further finds it.
        let baked = run(&low, &high, &settings(16, 1.5, 0.5));
        assert_eq!(baked.misses, 0);
    }

    #[test]
    fn occlusion_under_a_ceiling_matches_the_closed_form() {
        // Under a ceiling at height h, a cosine-weighted ray at angle θ from
        // the normal hits it at distance h / cos θ, so within d when
        // cos θ >= h / d; that happens with probability 1 - (h / d)². The
        // visibility is therefore (h / d)².
        let low = BakeMesh::from_retopo(&uv_grid(4, false));
        for (h, d) in [(0.5, 1.0), (0.3, 0.5), (0.9, 1.0)] {
            let floor = sheet(-50.0, 54.0, 26, false, |_, _| 0.0);
            let ceiling = sheet(-50.0, 54.0, 26, true, |_, _| h);
            let high = merge(&[floor, ceiling]);
            let mut settings = settings(32, 0.1, 0.1);
            settings.ao_distance = d;
            settings.ao_samples = 64;
            let baked = run(&low, &high, &settings);
            assert_eq!(baked.misses, 0);
            let ao = baked.ambient_occlusion.unwrap();
            let expected = (h / d) * (h / d);
            let mean = ao.data.iter().sum::<f32>() / ao.data.len() as f32;
            assert!(
                (mean - expected).abs() < 0.01,
                "h {h}, d {d}: {mean} vs {expected}"
            );
            // Stratified samples keep each texel close too.
            assert!(ao.data.iter().all(|&v| (v - expected).abs() < 0.1));
        }
        // Nothing above: fully visible.
        let high = sheet(-50.0, 54.0, 26, false, |_, _| 0.0);
        let baked = run(&low, &high, &settings(16, 0.1, 0.1));
        assert!(
            baked
                .ambient_occlusion
                .unwrap()
                .data
                .iter()
                .all(|&v| v == 1.0)
        );
    }

    #[test]
    fn bakes_are_deterministic_and_cancellable() {
        let low = BakeMesh::from_retopo(&uv_grid(3, false));
        let high = merge(&[
            sheet(-1.0, 4.0, 20, false, |x, y| {
                0.2 * (x * 3.0).sin() * (y * 2.0).cos()
            }),
            sheet(-1.0, 4.0, 4, true, |_, _| 0.6),
        ]);
        let settings = settings(24, 0.5, 0.5);
        assert_eq!(run(&low, &high, &settings), run(&low, &high, &settings));

        let bvh = Bvh::build(&high);
        let control = BakeControl::default();
        control.cancel();
        let reference = Reference {
            mesh: &high,
            bvh: &bvh,
        };
        assert_eq!(
            bake(&low, reference, &settings, &control),
            Err(BakeError::Cancelled)
        );
    }
}
