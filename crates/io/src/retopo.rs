//! Saving and loading retopology meshes as polygon OBJ files.
//!
//! Unlike reference meshes, retopology meshes keep their polygons (quads
//! stay quads) and are small, so this parser is sequential. Per-corner UVs
//! (`vt`) are kept; normals are ignored.

use std::io::Write;
use std::path::Path;

use std::collections::HashMap;

use polyflow_core::glam::{Vec2, Vec3};
use polyflow_core::halfedge::{HalfEdgeMesh, VertexId};

use crate::LoadError;
use crate::obj::{Cursor, ObjError, parse_int};

/// Writes the mesh as OBJ with its polygons and UVs. Vertex ids are
/// renumbered densely, so gaps left by deleted vertices do not reach the
/// file. Identical UVs are written once; faces with a corner lacking a UV are
/// written without any.
pub fn write_obj(mesh: &HalfEdgeMesh, out: impl Write) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(out);
    writeln!(
        out,
        "# Retopology exported by Polyflow {}",
        env!("CARGO_PKG_VERSION")
    )?;
    writeln!(
        out,
        "# {} vertices, {} faces",
        mesh.vertex_count(),
        mesh.face_count()
    )?;
    writeln!(out, "o retopo")?;
    let mut number = HashMap::with_capacity(mesh.vertex_count());
    for (i, v) in mesh.vertices().enumerate() {
        let p = mesh.position(v);
        writeln!(out, "v {} {} {}", p.x, p.y, p.z)?;
        number.insert(v, i + 1);
    }
    let mut uv_number: HashMap<[u32; 2], usize> = HashMap::new();
    let mut uv_values: Vec<Vec2> = Vec::new();
    let face_uvs: Vec<Option<Vec<usize>>> = mesh
        .faces()
        .map(|f| {
            let uvs = mesh.face_uvs(f)?;
            let numbers = uvs.iter().map(|&uv| {
                *uv_number
                    .entry(uv.to_array().map(f32::to_bits))
                    .or_insert_with(|| {
                        uv_values.push(uv);
                        uv_values.len()
                    })
            });
            Some(numbers.collect())
        })
        .collect();
    for uv in &uv_values {
        writeln!(out, "vt {} {}", uv.x, uv.y)?;
    }
    for (f, uvs) in mesh.faces().zip(face_uvs) {
        write!(out, "f")?;
        for (i, v) in mesh.face_vertices(f).enumerate() {
            match &uvs {
                Some(uvs) => write!(out, " {}/{}", number[&v], uvs[i])?,
                None => write!(out, " {}", number[&v])?,
            }
        }
        writeln!(out)?;
    }
    out.flush()
}

/// Saves to `path` atomically: the file is written next to it and renamed
/// into place, so a failure never leaves a truncated file behind.
pub fn save(path: &Path, mesh: &HalfEdgeMesh) -> std::io::Result<()> {
    crate::save_atomically(path, |file| write_obj(mesh, file))
}

/// A retopology mesh read from a file.
#[derive(Debug)]
pub struct ImportedRetopo {
    pub mesh: HalfEdgeMesh,
    /// Faces that could not be added: fewer than 3 distinct vertices, or
    /// they would make the mesh non-manifold.
    pub skipped_faces: usize,
}

pub fn load(path: &Path) -> Result<ImportedRetopo, LoadError> {
    let bytes = std::fs::read(path)?;
    Ok(parse_obj(&bytes)?)
}

/// Parses polygon OBJ data into a half-edge mesh. Never panics on malformed
/// input.
pub fn parse_obj(bytes: &[u8]) -> Result<ImportedRetopo, ObjError> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let mut positions: Vec<Vec3> = Vec::new();
    let mut uvs: Vec<Vec2> = Vec::new();
    // Faces stored flat: corner vertex indices, the matching UV indices
    // (`NO_UV` for faces without UVs), and where each face starts.
    const NO_UV: u32 = u32::MAX;
    let mut corners: Vec<u32> = Vec::new();
    let mut corner_uvs: Vec<u32> = Vec::new();
    let mut face_starts: Vec<usize> = Vec::new();
    let mut skipped_faces = 0;

    for (line_index, line) in bytes.split(|&b| b == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let error = |message: String| ObjError::Syntax {
            line: line_index + 1,
            message,
        };
        match line {
            [b'v', b' ' | b'\t', rest @ ..] => {
                let mut cursor = Cursor::new(rest);
                let mut p = [0.0; 3];
                for value in &mut p {
                    *value = cursor
                        .float()
                        .ok_or_else(|| error("expected three numbers after `v`".into()))?;
                }
                positions.push(Vec3::from_array(p));
            }
            [b'v', b't', b' ' | b'\t', rest @ ..] => {
                let mut cursor = Cursor::new(rest);
                let u = cursor
                    .float()
                    .ok_or_else(|| error("expected a number after `vt`".into()))?;
                // `v` is optional in the spec and defaults to 0.
                let v = cursor.float().unwrap_or(0.0);
                uvs.push(Vec2::new(u, v));
            }
            [b'f', b' ' | b'\t', rest @ ..] => {
                let start = corners.len();
                let mut cursor = Cursor::new(rest);
                let mut has_uvs = true;
                while let Some(token) = cursor.token() {
                    let mut parts = token.split(|&b| b == b'/');
                    let position = parts.next().unwrap_or_default();
                    corners.push(resolve(position, positions.len(), "vertex").map_err(error)?);
                    // A bad UV reference loses the face's UVs, not the face.
                    let uv = parts
                        .next()
                        .filter(|uv| !uv.is_empty())
                        .and_then(|uv| resolve(uv, uvs.len(), "UV").ok());
                    has_uvs &= uv.is_some();
                    corner_uvs.push(uv.unwrap_or(NO_UV));
                }
                if !has_uvs {
                    corner_uvs[start..].fill(NO_UV);
                }
                let face = &corners[start..];
                // Polygons are small: a quadratic check beats sorting a copy.
                let repeated = face
                    .iter()
                    .enumerate()
                    .any(|(i, v)| face[i + 1..].contains(v));
                if face.len() < 3 || repeated {
                    skipped_faces += 1;
                    corners.truncate(start);
                    corner_uvs.truncate(start);
                } else {
                    face_starts.push(start);
                }
            }
            _ => {}
        }
    }
    face_starts.push(corners.len());

    let mut mesh = HalfEdgeMesh::new();
    let ids: Vec<VertexId> = positions.iter().map(|&p| mesh.add_vertex(p)).collect();
    let mut face_vertices: Vec<VertexId> = Vec::new();
    let mut face_uvs: Vec<Vec2> = Vec::new();
    let mut add = |mesh: &mut HalfEdgeMesh, k: usize| {
        let range = face_starts[k]..face_starts[k + 1];
        face_vertices.clear();
        face_vertices.extend(corners[range.clone()].iter().map(|&i| ids[i as usize]));
        let Ok(f) = mesh.add_face(&face_vertices) else {
            return false;
        };
        if corner_uvs[range.start] != NO_UV {
            face_uvs.clear();
            face_uvs.extend(corner_uvs[range].iter().map(|&i| uvs[i as usize]));
            mesh.set_face_uvs(f, &face_uvs);
        }
        true
    };
    // Whether a face can be added may depend on the order: retry the ones
    // that failed until no more succeed.
    let mut pending: Vec<usize> = (0..face_starts.len() - 1)
        .filter(|&k| !add(&mut mesh, k))
        .collect();
    while !pending.is_empty() {
        let before = pending.len();
        pending.retain(|&k| !add(&mut mesh, k));
        if pending.len() == before {
            break;
        }
    }
    skipped_faces += pending.len();
    Ok(ImportedRetopo {
        mesh,
        skipped_faces,
    })
}

/// Turns a 1-based (or negative, relative) OBJ index into a 0-based one
/// below `len`.
fn resolve(token: &[u8], len: usize, what: &str) -> Result<u32, String> {
    let index = parse_int(token).ok_or_else(|| format!("invalid {what} index"))?;
    let resolved = match index {
        1.. => index - 1,
        ..0 => len as i64 + index,
        0 => {
            return Err(format!(
                "{what} index 0 is invalid (OBJ indices start at 1)"
            ));
        }
    };
    if resolved < 0 || resolved >= len as i64 {
        return Err(format!(
            "{what} index {index} refers to a {what} not defined yet"
        ));
    }
    Ok(resolved as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> HalfEdgeMesh {
        let mut mesh = HalfEdgeMesh::new();
        let mut ids = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                ids.push(mesh.add_vertex(Vec3::new(i as f32 * 0.1, j as f32 * 0.37, -1.25)));
            }
        }
        for j in 0..n {
            for i in 0..n {
                let k = j * (n + 1) + i;
                mesh.add_face(&[ids[k], ids[k + 1], ids[k + n + 2], ids[k + n + 1]])
                    .unwrap();
            }
        }
        mesh
    }

    #[test]
    fn round_trip_keeps_polygons_and_positions() {
        let mut mesh = grid(4);
        // Leave holes in the vertex ids, as deleting does.
        let corner = mesh.vertices().next().unwrap();
        mesh.remove_vertex(corner).unwrap();
        let mut text = Vec::new();
        write_obj(&mesh, &mut text).unwrap();
        let imported = parse_obj(&text).unwrap();
        assert_eq!(imported.skipped_faces, 0);
        let back = imported.mesh;
        back.validate().unwrap();
        assert_eq!(back.face_count(), mesh.face_count());
        assert_eq!(back.vertex_count(), mesh.vertex_count());
        assert!(
            back.faces().all(|f| back.face_halfedges(f).count() == 4),
            "quads stay quads"
        );
        let mut a: Vec<[u32; 3]> = mesh
            .vertices()
            .map(|v| mesh.position(v).to_array().map(f32::to_bits))
            .collect();
        let mut b: Vec<[u32; 3]> = back
            .vertices()
            .map(|v| back.position(v).to_array().map(f32::to_bits))
            .collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "positions survive exactly");
    }

    #[test]
    fn uvs_round_trip_with_seams() {
        let mut mesh = grid(2);
        let faces: Vec<_> = mesh.faces().collect();
        // Faces 0 and 1 share an edge but sit apart in UV space (a seam);
        // the other faces have no UVs.
        for (k, &f) in faces[..2].iter().enumerate() {
            let offset = Vec2::new(k as f32 * 0.6, 0.1);
            let uvs: Vec<Vec2> = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
                .iter()
                .map(|&(u, v)| Vec2::new(u, v) + offset)
                .collect();
            mesh.set_face_uvs(f, &uvs);
        }
        let mut text = Vec::new();
        write_obj(&mesh, &mut text).unwrap();
        let written = String::from_utf8(text.clone()).unwrap();
        assert_eq!(written.matches("\nvt ").count(), 8, "{written}");

        let back = parse_obj(&text).unwrap().mesh;
        back.validate().unwrap();
        // Faces with UVs, keyed by their vertex positions.
        let corners = |m: &HalfEdgeMesh| {
            let mut out: Vec<Vec<([u32; 3], [u32; 2])>> = m
                .faces()
                .filter_map(|f| {
                    let uvs = m.face_uvs(f)?;
                    let mut corners: Vec<_> = m
                        .face_vertices(f)
                        .zip(uvs)
                        .map(|(v, uv)| {
                            (
                                m.position(v).to_array().map(f32::to_bits),
                                uv.to_array().map(f32::to_bits),
                            )
                        })
                        .collect();
                    corners.sort_unstable();
                    Some(corners)
                })
                .collect();
            out.sort_unstable();
            out
        };
        assert_eq!(corners(&back).len(), 2);
        assert_eq!(corners(&back), corners(&mesh));
    }

    #[test]
    fn uv_indices_are_parsed_leniently() {
        let text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 0.25\n\
                    f 1/1/1 2/2/1 3/-1/1\nf 2/2 4/9 3/3\n";
        let mesh = parse_obj(text.as_bytes()).unwrap().mesh;
        let mut faces = mesh.faces();
        let first = faces.next().unwrap();
        assert_eq!(
            mesh.face_uvs(first).unwrap(),
            [Vec2::ZERO, Vec2::X, Vec2::new(0.25, 0.0)],
            "relative indices and a missing `v` component"
        );
        let second = faces.next().unwrap();
        assert_eq!(
            mesh.face_uvs(second),
            None,
            "an undefined UV drops the face's UVs"
        );
    }

    #[test]
    fn faces_in_awkward_orders_are_retried() {
        // Ring around a center written in an order that needs retries.
        let mut text = String::from("v 0 0 0\n");
        for k in 0..6 {
            let a = k as f32 / 6.0 * std::f32::consts::TAU;
            text += &format!("v {} {} 0\n", a.cos(), a.sin());
        }
        for k in [0, 3, 1, 4, 2, 5] {
            text += &format!("f 1 {} {}\n", k + 2, (k + 1) % 6 + 2);
        }
        let imported = parse_obj(text.as_bytes()).unwrap();
        assert_eq!(imported.skipped_faces, 0);
        assert_eq!(imported.mesh.face_count(), 6);
        imported.mesh.validate().unwrap();
    }

    #[test]
    fn bad_faces_are_skipped_and_errors_reported() {
        let text =
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 1 2 3\nf 1 1 2\nf 4 3/1/1 2//2\n";
        let imported = parse_obj(text.as_bytes()).unwrap();
        // The duplicate face and the repeated-vertex face are skipped.
        assert_eq!((imported.mesh.face_count(), imported.skipped_faces), (2, 2));
        assert!(matches!(
            parse_obj(b"v 0 0 0\nf 1 2 3\n"),
            Err(ObjError::Syntax { line: 2, .. })
        ));
        assert!(matches!(
            parse_obj(b"v 0 x 0\n"),
            Err(ObjError::Syntax { line: 1, .. })
        ));
    }

    #[test]
    fn save_is_atomic_and_replaces() {
        let dir = std::env::temp_dir().join(format!("polyflow-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("retopo.obj");
        save(&path, &grid(2)).unwrap();
        save(&path, &grid(3)).unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.mesh.face_count(), 9);
        assert!(!dir.join("retopo.obj.tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn garbage_never_panics() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let alphabet = b"vf 0123-/.\n#";
        for _ in 0..2000 {
            let len = (state % 120) as usize;
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    alphabet[(state % alphabet.len() as u64) as usize]
                })
                .collect();
            if let Ok(imported) = parse_obj(&bytes) {
                imported.mesh.validate().unwrap();
            }
        }
    }
}
