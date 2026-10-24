//! MikkTSpace tangents and the tangent-space encoding of normals.
//!
//! A MikkTSpace shader interpolates the vertex normal `N` and tangent `T`
//! without normalizing them, builds the bitangent per pixel as
//! `B = sign * cross(N, T)`, and lights with
//! `normalize(n.x * T + n.y * B + n.z * N)` for the tangent-space normal `n`
//! read from the map. Baking inverts exactly that, so the shader gets back
//! the baked normal even where the basis is not orthogonal.

use bevy_mikktspace::{Geometry, TangentSpace};
use polyflow_core::glam::{Mat3, Vec3, Vec4};

use crate::mesh::Corner;

/// The tangent basis at a point, as a MikkTSpace shader sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TangentFrame {
    /// Interpolated vertex normal, not normalized.
    pub normal: Vec3,
    /// Interpolated tangent, not normalized.
    pub tangent: Vec3,
    /// Bitangent sign: -1 where the UVs are mirrored.
    pub sign: f32,
}

impl TangentFrame {
    pub fn bitangent(&self) -> Vec3 {
        self.sign * self.normal.cross(self.tangent)
    }

    fn matrix(&self) -> Mat3 {
        Mat3::from_cols(self.tangent, self.bitangent(), self.normal)
    }

    /// The tangent-space normal (unit length) that a MikkTSpace shader turns
    /// back into `world_normal`'s direction.
    pub fn encode(&self, world_normal: Vec3) -> Vec3 {
        let m = self.matrix();
        // |T| |B| |N| = |T|² |N|²: the determinant relative to that measures
        // how far the basis is from flat.
        let scale = self.normal.length_squared() * self.tangent.length_squared();
        // A degenerate basis (zero tangent, tangent along the normal): fall
        // back to an orthonormal one so the result is still sensible.
        let local = if m.determinant().abs() > 1e-6 * scale {
            m.inverse() * world_normal
        } else {
            let n = self.normal.normalize_or(Vec3::Z);
            let t = self
                .tangent
                .reject_from(n)
                .normalize_or(n.any_orthonormal_vector());
            let b = self.sign * n.cross(t);
            Vec3::new(
                world_normal.dot(t),
                world_normal.dot(b),
                world_normal.dot(n),
            )
        };
        local.normalize_or(Vec3::Z)
    }

    /// What a MikkTSpace shader computes from a tangent-space normal.
    pub fn decode(&self, tangent_normal: Vec3) -> Vec3 {
        (self.matrix() * tangent_normal).normalize_or(self.normal)
    }
}

/// The tangent the MikkTSpace reference reports where it cannot compute one.
const DEFAULT_TANGENT: Vec4 = Vec4::new(1.0, 0.0, 0.0, -1.0);

/// Fills in `corners[i].tangent` (three corners per triangle) with
/// MikkTSpace tangents.
pub fn generate(corners: &mut [Corner]) {
    struct Triangles<'a>(&'a mut [Corner]);

    impl Geometry for Triangles<'_> {
        fn num_faces(&self) -> usize {
            self.0.len() / 3
        }
        fn num_vertices_of_face(&self, _face: usize) -> usize {
            3
        }
        fn position(&self, face: usize, vert: usize) -> [f32; 3] {
            self.0[face * 3 + vert].position.to_array()
        }
        fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
            self.0[face * 3 + vert].normal.to_array()
        }
        fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
            self.0[face * 3 + vert].uv.to_array()
        }
        fn set_tangent(&mut self, tangent: Option<TangentSpace>, face: usize, vert: usize) {
            let corner = &mut self.0[face * 3 + vert];
            corner.tangent = match tangent {
                Some(t) => Vec4::from_array(t.tangent_encoded()),
                // MikkTSpace found nothing to derive a tangent from (a face
                // with degenerate UVs and no usable neighbors): write what
                // the reference C code leaves there, +X with a -1 sign.
                None => DEFAULT_TANGENT,
            };
        }
    }

    // The error type has no variants yet; should one appear, fall back as
    // the reference does when it fails.
    if bevy_mikktspace::generate_tangents(&mut Triangles(corners)).is_err() {
        for corner in corners {
            corner.tangent = DEFAULT_TANGENT;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic PRNG.
    struct Rng(u64);
    impl Rng {
        fn unit(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
        fn vec(&mut self) -> Vec3 {
            Vec3::new(self.unit(), self.unit(), self.unit())
        }
    }

    #[test]
    fn encoding_inverts_the_shader_on_skewed_frames() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for _ in 0..10_000 {
            // Interpolated vectors: neither unit length nor orthogonal.
            let normal = (Vec3::Z + rng.vec() * 0.3) * (0.7 + rng.unit().abs() * 0.3);
            let tangent = (Vec3::X + rng.vec() * 0.3) * (0.7 + rng.unit().abs() * 0.3);
            let frame = TangentFrame {
                normal,
                tangent,
                sign: if rng.unit() < 0.0 { -1.0 } else { 1.0 },
            };
            // Detail normals stay within the front hemisphere.
            let world = (normal.normalize() + rng.vec() * 0.6).normalize();
            let encoded = frame.encode(world);
            assert!((encoded.length() - 1.0).abs() < 1e-5);
            let decoded = frame.decode(encoded);
            assert!(
                decoded.distance(world) < 1e-4,
                "{frame:?}: {world} -> {encoded} -> {decoded}"
            );
        }
    }

    /// A torus of quads (closed in both directions, so UVs have seams both
    /// ways) whose second half reuses the first half's UVs mirrored.
    fn mirrored_torus(rings: usize, sides: usize) -> polyflow_core::halfedge::HalfEdgeMesh {
        use polyflow_core::glam::Vec2;
        use std::f32::consts::TAU;
        let mut mesh = polyflow_core::halfedge::HalfEdgeMesh::new();
        let mut ids = Vec::new();
        for i in 0..rings {
            for j in 0..sides {
                let (a, b) = (i as f32 / rings as f32 * TAU, j as f32 / sides as f32 * TAU);
                let r = 1.0 + 0.35 * b.cos();
                ids.push(mesh.add_vertex(Vec3::new(r * a.cos(), r * a.sin(), 0.35 * b.sin())));
            }
        }
        let id = |i: usize, j: usize| ids[(i % rings) * sides + j % sides];
        let uv = |i: usize, j: usize| {
            let half = rings / 2;
            let u = if i <= half { i } else { rings - i };
            Vec2::new(u as f32 / half as f32, j as f32 / sides as f32)
        };
        for i in 0..rings {
            for j in 0..sides {
                let f = mesh
                    .add_face(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)])
                    .unwrap();
                mesh.set_face_uvs(f, &[uv(i, j), uv(i + 1, j), uv(i + 1, j + 1), uv(i, j + 1)]);
            }
        }
        mesh
    }

    #[test]
    fn tangents_match_the_reference_c_code_bit_for_bit() {
        struct Reference<'a>(&'a [crate::mesh::Corner], Vec<[f32; 4]>);
        impl mikktspace_sys::MikkTSpaceInterface for Reference<'_> {
            fn get_num_faces(&self) -> usize {
                self.0.len() / 3
            }
            fn get_num_vertices_of_face(&self, _face: usize) -> usize {
                3
            }
            fn get_position(&self, face: usize, vert: usize) -> [f32; 3] {
                self.0[face * 3 + vert].position.to_array()
            }
            fn get_normal(&self, face: usize, vert: usize) -> [f32; 3] {
                self.0[face * 3 + vert].normal.to_array()
            }
            fn get_tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
                self.0[face * 3 + vert].uv.to_array()
            }
            fn set_tspace_basic(&mut self, tangent: [f32; 3], sign: f32, face: usize, vert: usize) {
                self.1[face * 3 + vert] = [tangent[0], tangent[1], tangent[2], sign];
            }
        }

        for (rings, sides) in [(8, 4), (24, 12), (61, 17)] {
            let mesh = crate::BakeMesh::from_retopo(&mirrored_torus(rings, sides));
            let mut reference = Reference(&mesh.corners, vec![[0.0; 4]; mesh.corners.len()]);
            assert!(mikktspace_sys::gen_tang_space_default(&mut reference));
            let mut mirrored = 0;
            for (i, (corner, expected)) in mesh.corners.iter().zip(&reference.1).enumerate() {
                let ours = corner.tangent.to_array().map(f32::to_bits);
                assert_eq!(
                    ours,
                    expected.map(f32::to_bits),
                    "corner {i} of torus {rings}x{sides}"
                );
                mirrored += (corner.tangent.w < 0.0) as usize;
            }
            assert!(
                mirrored > 0 && mirrored < mesh.corners.len(),
                "both orientations are tested"
            );
        }
    }

    #[test]
    fn the_vertex_normal_encodes_as_flat() {
        let frame = TangentFrame {
            normal: Vec3::new(0.1, 0.2, 0.9),
            tangent: Vec3::new(1.0, 0.0, -0.1),
            sign: -1.0,
        };
        let flat = frame.encode(frame.normal.normalize());
        assert!(flat.distance(Vec3::Z) < 1e-5, "{flat}");
    }

    #[test]
    fn degenerate_frames_still_encode() {
        let frame = TangentFrame {
            normal: Vec3::Z,
            tangent: Vec3::ZERO,
            sign: 1.0,
        };
        let encoded = frame.encode(Vec3::new(0.0, 0.6, 0.8));
        assert!((encoded.length() - 1.0).abs() < 1e-5);
        assert!((encoded.z - 0.8).abs() < 1e-5);
    }
}
