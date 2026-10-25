//! Writing the low-poly exactly as it was baked, for game engines.
//!
//! A normal map only shades correctly on the triangles, normals and UVs it
//! was baked for, and engines rebuild MikkTSpace tangents from those. So the
//! bake also exports its own triangulation with its normals, instead of the
//! polygons of the retopology: as OBJ here, and as glTF with the tangents
//! too in [`crate::gltf::write_glb`].

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use polyflow_core::glam::{Vec2, Vec3, Vec4};

/// One corner of a triangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshCorner {
    pub position: Vec3,
    pub normal: Vec3,
    /// With V up (OBJ's convention).
    pub uv: Vec2,
    /// MikkTSpace tangent with its handedness in `w` (OBJ has no place
    /// for it; glTF does).
    pub tangent: Vec4,
}

/// Writes triangles (three corners each) as OBJ with UVs and normals.
/// Corners with bitwise identical positions share a vertex, so the mesh
/// stays connected; UVs and normals are shared the same way.
pub fn write_obj(corners: &[MeshCorner], out: impl Write) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(out);
    writeln!(
        out,
        "# Low-poly as baked by Polyflow {}: use these triangles and normals with the maps",
        env!("CARGO_PKG_VERSION")
    )?;
    writeln!(out, "# {} triangles", corners.len() / 3)?;
    writeln!(out, "o lowpoly")?;

    // Numbers (1-based) of the distinct positions, UVs and normals.
    fn number<const N: usize>(
        map: &mut HashMap<[u32; N], usize>,
        list: &mut Vec<[f32; N]>,
        value: [f32; N],
    ) -> usize {
        *map.entry(value.map(f32::to_bits)).or_insert_with(|| {
            list.push(value);
            list.len()
        })
    }
    let (mut positions, mut uvs, mut normals) = (Vec::new(), Vec::new(), Vec::new());
    let (mut p_map, mut t_map, mut n_map) = (HashMap::new(), HashMap::new(), HashMap::new());
    let indices: Vec<[usize; 3]> = corners
        .iter()
        .map(|c| {
            [
                number(&mut p_map, &mut positions, c.position.to_array()),
                number(&mut t_map, &mut uvs, c.uv.to_array()),
                number(&mut n_map, &mut normals, c.normal.to_array()),
            ]
        })
        .collect();
    for [x, y, z] in &positions {
        writeln!(out, "v {x} {y} {z}")?;
    }
    for [u, v] in &uvs {
        writeln!(out, "vt {u} {v}")?;
    }
    for [x, y, z] in &normals {
        writeln!(out, "vn {x} {y} {z}")?;
    }
    for triangle in indices.as_chunks::<3>().0 {
        write!(out, "f")?;
        for [p, t, n] in triangle {
            write!(out, " {p}/{t}/{n}")?;
        }
        writeln!(out)?;
    }
    out.flush()
}

/// [`write_obj`] to a file, atomically (see [`crate::retopo::save`]).
pub fn save_obj(path: &Path, corners: &[MeshCorner]) -> std::io::Result<()> {
    crate::save_atomically(path, |file| write_obj(corners, file))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_values_are_written_once() {
        // Two triangles of a quad sharing an edge; a UV seam on one corner.
        let corner = |x: f32, y: f32, u: f32| MeshCorner {
            position: Vec3::new(x, y, 0.0),
            normal: Vec3::Z,
            uv: Vec2::new(u, y),
            tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
        };
        let corners = [
            corner(0.0, 0.0, 0.0),
            corner(1.0, 0.0, 1.0),
            corner(1.0, 1.0, 1.0),
            corner(0.0, 0.0, 0.0),
            corner(1.0, 1.0, 0.5), // same position, another UV
            corner(0.0, 1.0, 0.0),
        ];
        let mut text = Vec::new();
        write_obj(&corners, &mut text).unwrap();
        let text = String::from_utf8(text).unwrap();
        let count = |prefix: &str| text.lines().filter(|l| l.starts_with(prefix)).count();
        assert_eq!(
            (count("v "), count("vt "), count("vn "), count("f ")),
            (4, 5, 1, 2)
        );
        assert!(
            text.contains("\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/4/1 4/5/1\n"),
            "{text}"
        );

        // Reads back as a connected mesh with its UVs.
        let back = crate::retopo::parse_obj(text.as_bytes()).unwrap();
        assert_eq!((back.mesh.vertex_count(), back.mesh.face_count()), (4, 2));
        back.mesh.validate().unwrap();
        assert!(back.mesh.faces().all(|f| back.mesh.face_uvs(f).is_some()));
    }
}
