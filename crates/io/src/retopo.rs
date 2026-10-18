//! Saving and loading retopology meshes as polygon OBJ files.
//!
//! Unlike reference meshes, retopology meshes keep their polygons (quads
//! stay quads) and are small, so this parser is sequential.

use std::io::Write;
use std::path::Path;

use polyflow_core::glam::Vec3;
use polyflow_core::halfedge::{HalfEdgeMesh, VertexId};

use crate::LoadError;
use crate::obj::{Cursor, ObjError, parse_int};

/// Writes the mesh as OBJ with its polygons. Vertex ids are renumbered
/// densely, so gaps left by deleted vertices do not reach the file.
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
    let mut number = std::collections::HashMap::with_capacity(mesh.vertex_count());
    for (i, v) in mesh.vertices().enumerate() {
        let p = mesh.position(v);
        writeln!(out, "v {} {} {}", p.x, p.y, p.z)?;
        number.insert(v, i + 1);
    }
    for f in mesh.faces() {
        write!(out, "f")?;
        for v in mesh.face_vertices(f) {
            write!(out, " {}", number[&v])?;
        }
        writeln!(out)?;
    }
    out.flush()
}

/// Saves to `path` atomically: the file is written next to it and renamed
/// into place, so a failure never leaves a truncated file behind.
pub fn save(path: &Path, mesh: &HalfEdgeMesh) -> std::io::Result<()> {
    let mut temp = path.as_os_str().to_owned();
    temp.push(".tmp");
    let temp = std::path::PathBuf::from(temp);
    let result = std::fs::File::create(&temp).and_then(|file| {
        write_obj(mesh, &file)?;
        file.sync_all()
    });
    match result.and_then(|()| std::fs::rename(&temp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&temp);
            Err(e)
        }
    }
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
    let mut faces: Vec<Vec<u32>> = Vec::new();
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
            [b'f', b' ' | b'\t', rest @ ..] => {
                let mut cursor = Cursor::new(rest);
                let mut face = Vec::new();
                while let Some(token) = cursor.token() {
                    let position = token.split(|&b| b == b'/').next().unwrap_or_default();
                    let index =
                        parse_int(position).ok_or_else(|| error("invalid face index".into()))?;
                    let resolved = match index {
                        1.. => index - 1,
                        ..0 => positions.len() as i64 + index,
                        0 => {
                            return Err(error(
                                "face index 0 is invalid (OBJ indices start at 1)".into(),
                            ));
                        }
                    };
                    if resolved < 0 || resolved >= positions.len() as i64 {
                        return Err(error(format!(
                            "face index {index} refers to a vertex not defined yet"
                        )));
                    }
                    face.push(resolved as u32);
                }
                let mut unique = face.clone();
                unique.sort_unstable();
                unique.dedup();
                if face.len() < 3 || unique.len() != face.len() {
                    skipped_faces += 1;
                } else {
                    faces.push(face);
                }
            }
            _ => {}
        }
    }

    let mut mesh = HalfEdgeMesh::new();
    let ids: Vec<VertexId> = positions.iter().map(|&p| mesh.add_vertex(p)).collect();
    // Whether a face can be added may depend on the order: retry the ones
    // that failed until no more succeed.
    let mut pending: Vec<Vec<VertexId>> = faces
        .iter()
        .map(|f| f.iter().map(|&i| ids[i as usize]).collect())
        .collect();
    loop {
        let before = pending.len();
        pending.retain(|face| mesh.add_face(face).is_err());
        if pending.is_empty() || pending.len() == before {
            break;
        }
    }
    skipped_faces += pending.len();
    Ok(ImportedRetopo {
        mesh,
        skipped_faces,
    })
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
