//! Wavefront OBJ import and export.
//!
//! The importer is built for multi-gigabyte sculpt exports: the file is split
//! into chunks at line boundaries and parsed on all cores. Only geometry is
//! read (`v` and `f`); polygons are fan-triangulated and normals are
//! recomputed, since a reference mesh needs one smooth normal per position.
//! Everything else (`vt`, `vn`, groups, materials, ...) is ignored for now.

use std::io::Write;
use std::ops::Range;

use polyflow_core::glam::Vec3;
use polyflow_core::mesh::TriMesh;
use rayon::prelude::*;

/// Target chunk size for parallel parsing.
const CHUNK_BYTES: usize = 4 << 20;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ObjError {
    #[error("line {line}: {message}")]
    Syntax { line: usize, message: String },
    #[error("face references vertex {index}, but the file has {count} vertices")]
    IndexOutOfRange { index: i64, count: usize },
    #[error("the mesh is too large (more than 2^32 vertices or indices)")]
    TooLarge,
}

/// Geometry parsed from an OBJ file, before normals are computed.
#[derive(Debug, Default)]
pub struct ObjGeometry {
    pub positions: Vec<Vec3>,
    pub indices: Vec<u32>,
    /// Faces with fewer than three vertices, which were skipped.
    pub skipped_faces: usize,
}

impl ObjGeometry {
    /// Converts into a mesh, computing smooth vertex normals.
    pub fn into_mesh(self) -> TriMesh {
        let mut mesh = TriMesh {
            positions: self.positions,
            normals: vec![],
            indices: self.indices,
        };
        mesh.compute_normals();
        mesh
    }
}

/// Parses OBJ data. Never panics on malformed input.
pub fn parse(bytes: &[u8]) -> Result<ObjGeometry, ObjError> {
    parse_with_chunk_size(bytes, CHUNK_BYTES)
}

fn parse_with_chunk_size(bytes: &[u8], chunk_bytes: usize) -> Result<ObjGeometry, ObjError> {
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(bytes);
    let ranges = split_at_lines(bytes, chunk_bytes);
    let chunks: Vec<Chunk> = ranges
        .par_iter()
        .map(|range| {
            parse_chunk(&bytes[range.clone()]).map_err(|(offset, message)| ObjError::Syntax {
                line: line_number(bytes, range.start + offset),
                message,
            })
        })
        .collect::<Result<_, _>>()?;
    merge(chunks)
}

/// Splits `bytes` into ranges of roughly `chunk_bytes`, each ending after a newline.
fn split_at_lines(bytes: &[u8], chunk_bytes: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::with_capacity(bytes.len() / chunk_bytes + 1);
    let mut start = 0;
    while start < bytes.len() {
        let target = (start + chunk_bytes).min(bytes.len());
        let end = match memchr::memchr(b'\n', &bytes[target..]) {
            Some(i) => target + i + 1,
            None => bytes.len(),
        };
        ranges.push(start..end);
        start = end;
    }
    ranges
}

fn line_number(bytes: &[u8], offset: usize) -> usize {
    memchr::memchr_iter(b'\n', &bytes[..offset.min(bytes.len())]).count() + 1
}

#[derive(Default)]
struct Chunk {
    positions: Vec<Vec3>,
    /// Global 0-based indices. Entries listed in `relative` are placeholders.
    indices: Vec<u32>,
    /// `(position in indices, vertex index relative to the chunk start)` for
    /// faces using negative OBJ indices, whose global value depends on how many
    /// vertices earlier chunks define. Rare in practice.
    relative: Vec<(usize, i64)>,
    skipped_faces: usize,
}

#[derive(Clone, Copy)]
enum Corner {
    Global(u32),
    Relative(i64),
}

/// Parses one chunk of whole lines. Errors carry the byte offset in the chunk.
fn parse_chunk(bytes: &[u8]) -> Result<Chunk, (usize, String)> {
    let mut chunk = Chunk::default();
    let mut corners: Vec<Corner> = Vec::with_capacity(16);
    let mut line_start = 0;
    while line_start < bytes.len() {
        let line_end =
            memchr::memchr(b'\n', &bytes[line_start..]).map_or(bytes.len(), |i| line_start + i);
        let line = &bytes[line_start..line_end];
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        let error = |message: String| (line_start, message);

        if line.ends_with(b"\\") {
            return Err(error("line continuations are not supported".into()));
        }
        match line {
            [b'v', b' ' | b'\t', rest @ ..] => {
                let mut cursor = Cursor::new(rest);
                let mut p = [0.0; 3];
                for value in &mut p {
                    *value = cursor
                        .float()
                        .ok_or_else(|| error("expected three numbers after `v`".into()))?;
                }
                // Extra values (w, or per-vertex colors) are ignored.
                chunk.positions.push(Vec3::from_array(p));
            }
            [b'f', b' ' | b'\t', rest @ ..] => {
                corners.clear();
                let mut cursor = Cursor::new(rest);
                while let Some(token) = cursor.token() {
                    // `v`, `v/vt`, `v//vn` or `v/vt/vn`: only the position matters.
                    let position = token.split(|&b| b == b'/').next().unwrap_or_default();
                    let index = parse_int(position)
                        .ok_or_else(|| error(format!("invalid face index `{}`", show(token))))?;
                    corners.push(match index {
                        1.. => {
                            Corner::Global(u32::try_from(index - 1).map_err(|_| {
                                error(format!("face index {index} is out of range"))
                            })?)
                        }
                        // -1 is the last vertex defined so far.
                        ..0 => Corner::Relative(chunk.positions.len() as i64 + index),
                        0 => {
                            return Err(error(
                                "face index 0 is invalid (OBJ indices start at 1)".into(),
                            ));
                        }
                    });
                }
                if corners.len() < 3 {
                    chunk.skipped_faces += 1;
                    continue_line(&mut line_start, line_end);
                    continue;
                }
                for i in 1..corners.len() - 1 {
                    for corner in [corners[0], corners[i], corners[i + 1]] {
                        match corner {
                            Corner::Global(index) => chunk.indices.push(index),
                            Corner::Relative(local) => {
                                chunk.relative.push((chunk.indices.len(), local));
                                chunk.indices.push(0);
                            }
                        }
                    }
                }
            }
            _ => {} // Comments, normals, UVs, groups, materials, ...
        }
        continue_line(&mut line_start, line_end);
    }
    Ok(chunk)
}

fn continue_line(line_start: &mut usize, line_end: usize) {
    *line_start = line_end + 1;
}

fn merge(chunks: Vec<Chunk>) -> Result<ObjGeometry, ObjError> {
    let mut vertex_offsets = Vec::with_capacity(chunks.len());
    let (mut vertex_count, mut index_count) = (0usize, 0usize);
    for chunk in &chunks {
        vertex_offsets.push(vertex_count);
        vertex_count += chunk.positions.len();
        index_count += chunk.indices.len();
    }
    if vertex_count > u32::MAX as usize || index_count > u32::MAX as usize {
        return Err(ObjError::TooLarge);
    }

    let mut positions = vec![Vec3::ZERO; vertex_count];
    let mut indices = vec![0u32; index_count];
    let position_slices = split_mut(&mut positions, chunks.iter().map(|c| c.positions.len()));
    let index_slices = split_mut(&mut indices, chunks.iter().map(|c| c.indices.len()));

    chunks
        .par_iter()
        .zip(position_slices)
        .zip(index_slices)
        .zip(&vertex_offsets)
        .try_for_each(|(((chunk, positions), indices), &vertex_offset)| {
            positions.copy_from_slice(&chunk.positions);
            indices.copy_from_slice(&chunk.indices);
            for &(at, local) in &chunk.relative {
                let global = vertex_offset as i64 + local;
                // Negative indices may only refer back to vertices defined so far.
                if global < 0 || global >= (vertex_offset + chunk.positions.len()) as i64 {
                    return Err(ObjError::IndexOutOfRange {
                        index: global + 1,
                        count: vertex_count,
                    });
                }
                indices[at] = global as u32;
            }
            if let Some(&bad) = indices.iter().find(|&&i| i as usize >= vertex_count) {
                return Err(ObjError::IndexOutOfRange {
                    index: i64::from(bad) + 1,
                    count: vertex_count,
                });
            }
            Ok(())
        })?;

    Ok(ObjGeometry {
        positions,
        indices,
        skipped_faces: chunks.iter().map(|c| c.skipped_faces).sum(),
    })
}

/// Splits `slice` into consecutive mutable pieces of the given lengths.
fn split_mut<T>(mut slice: &mut [T], lengths: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    lengths
        .map(|len| {
            let (head, tail) = std::mem::take(&mut slice).split_at_mut(len);
            slice = tail;
            head
        })
        .collect()
}

/// Whitespace-separated scanning within one line.
pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    fn skip_whitespace(&mut self) {
        let n = self
            .bytes
            .iter()
            .take_while(|b| matches!(b, b' ' | b'\t'))
            .count();
        self.bytes = &self.bytes[n..];
    }

    pub(crate) fn token(&mut self) -> Option<&'a [u8]> {
        self.skip_whitespace();
        if self.bytes.is_empty() || self.bytes[0] == b'#' {
            return None;
        }
        let n = self
            .bytes
            .iter()
            .take_while(|b| !matches!(b, b' ' | b'\t'))
            .count();
        let (token, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Some(token)
    }

    pub(crate) fn float(&mut self) -> Option<f32> {
        let token = self.token()?;
        match fast_float2::parse_partial::<f32, _>(token) {
            Ok((value, consumed)) if consumed == token.len() && value.is_finite() => Some(value),
            _ => None,
        }
    }
}

pub(crate) fn parse_int(bytes: &[u8]) -> Option<i64> {
    let (negative, digits) = match bytes {
        [b'-', rest @ ..] => (true, rest),
        [b'+', rest @ ..] => (false, rest),
        _ => (false, bytes),
    };
    if digits.is_empty() || digits.len() > 18 {
        return None;
    }
    let mut value: i64 = 0;
    for &d in digits {
        if !d.is_ascii_digit() {
            return None;
        }
        value = value * 10 + i64::from(d - b'0');
    }
    Some(if negative { -value } else { value })
}

fn show(bytes: &[u8]) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(32)]).into_owned()
}

/// Writes `mesh` as OBJ (positions and triangles only).
pub fn write(mesh: &TriMesh, out: impl Write) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::with_capacity(1 << 20, out);
    writeln!(out, "# Exported by Polyflow {}", env!("CARGO_PKG_VERSION"))?;
    for p in &mesh.positions {
        writeln!(out, "v {} {} {}", p.x, p.y, p.z)?;
    }
    for [a, b, c] in mesh.indices.as_chunks::<3>().0 {
        writeln!(out, "f {} {} {}", a + 1, b + 1, c + 1)?;
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use polyflow_core::primitives::torus;

    use super::*;

    fn parse_str(s: &str) -> Result<ObjGeometry, ObjError> {
        parse(s.as_bytes())
    }

    #[test]
    fn triangles_quads_and_polygons() {
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0.5 2 0\n\
                   f 1 2 3\nf 1 2 3 4\nf 1 2 3 5 4\n";
        let g = parse_str(obj).unwrap();
        assert_eq!(g.positions.len(), 5);
        assert_eq!(g.indices.len() / 3, 1 + 2 + 3);
        assert_eq!(&g.indices[3..9], &[0, 1, 2, 0, 2, 3]);
    }

    #[test]
    fn corner_formats_comments_and_crlf() {
        let obj = "# comment\r\nmtllib x.mtl\r\no thing\r\nv 0 0 0 1\r\nv 1e0 0 0\r\nv 0 +1 -0.0 0.5 0.5 0.5\r\n\
                   vt 0 0\r\nvn 0 0 1\r\ng grp\r\ns 1\r\nusemtl m\r\n\
                   f 1/1/1 2//1 3/1 # trailing comment\r\nf\t1 2\t3\r\n";
        let g = parse_str(obj).unwrap();
        assert_eq!(g.positions[1], Vec3::X);
        assert_eq!(g.positions[2], Vec3::Y);
        assert_eq!(g.indices, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn negative_indices() {
        let g = parse_str("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\nv 0 0 1\nf -4 -1 -2\n").unwrap();
        assert_eq!(g.indices, vec![0, 1, 2, 0, 3, 2]);
    }

    #[test]
    fn negative_indices_across_chunks() {
        let mut obj = String::new();
        for i in 0..200 {
            obj += &format!("v {i} 0 0\nv {i} 1 0\nv {i} 0 1\nf -3 -2 -1\n");
        }
        obj += "f -600 -1 1\n"; // reaches back into the first chunks
        let g = parse_with_chunk_size(obj.as_bytes(), 64).unwrap();
        for i in 0..200u32 {
            assert_eq!(
                &g.indices[i as usize * 3..i as usize * 3 + 3],
                &[3 * i, 3 * i + 1, 3 * i + 2]
            );
        }
        assert_eq!(&g.indices[600..], &[0, 599, 0]);
    }

    #[test]
    fn chunked_parse_matches_single_chunk() {
        let mesh = torus(1.0, 0.3, 37, 23);
        let mut obj = Vec::new();
        write(&mesh, &mut obj).unwrap();
        let whole = parse_with_chunk_size(&obj, usize::MAX / 2).unwrap();
        for chunk in [1, 7, 100, 4096] {
            let chunked = parse_with_chunk_size(&obj, chunk).unwrap();
            assert_eq!(chunked.positions, whole.positions);
            assert_eq!(chunked.indices, whole.indices);
        }
    }

    #[test]
    fn round_trip() {
        let mesh = torus(1.0, 0.3, 40, 20);
        let mut obj = Vec::new();
        write(&mesh, &mut obj).unwrap();
        let g = parse(&obj).unwrap();
        assert_eq!(
            g.positions, mesh.positions,
            "f32 Display round-trips exactly"
        );
        assert_eq!(g.indices, mesh.indices);
        let loaded = g.into_mesh();
        assert_eq!(loaded.normals.len(), loaded.positions.len());
    }

    #[test]
    fn degenerate_faces_are_skipped() {
        let g = parse_str("v 0 0 0\nv 1 0 0\nf 1 2\nf 1\n").unwrap();
        assert_eq!(g.skipped_faces, 2);
        assert!(g.indices.is_empty());
    }

    #[test]
    fn errors_report_line_numbers() {
        let err = parse_str("v 0 0 0\nv 1 0\n").unwrap_err();
        assert!(matches!(err, ObjError::Syntax { line: 2, .. }), "{err:?}");
        let err = parse_str("v 0 0 0\n\nf 1 x 1\n").unwrap_err();
        assert!(matches!(err, ObjError::Syntax { line: 3, .. }), "{err:?}");
        let err = parse_str("v 0 0 0\nf 0 1 1\n").unwrap_err();
        assert!(matches!(err, ObjError::Syntax { line: 2, .. }), "{err:?}");
        let err = parse_str("v nan 0 0\n").unwrap_err();
        assert!(matches!(err, ObjError::Syntax { line: 1, .. }), "{err:?}");
    }

    #[test]
    fn out_of_range_indices() {
        assert_eq!(
            parse_str("v 0 0 0\nf 1 2 3\n").unwrap_err(),
            ObjError::IndexOutOfRange { index: 2, count: 1 }
        );
        assert!(matches!(
            parse_str("v 0 0 0\nf -1 -2 -1\n").unwrap_err(),
            ObjError::IndexOutOfRange { .. }
        ));
        assert!(parse_str("v 0 0 0\nf 99999999999999999999 1 1\n").is_err());
    }

    #[test]
    fn garbage_never_panics() {
        // Small deterministic PRNG so the test needs no extra dependency.
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let alphabet = b"vf 0123456789-+./\n\r\t#eE\\nx";
        for _ in 0..2000 {
            let len = (next() % 200) as usize;
            let bytes: Vec<u8> = (0..len)
                .map(|_| alphabet[(next() % alphabet.len() as u64) as usize])
                .collect();
            for chunk in [3, 4096] {
                let _ = parse_with_chunk_size(&bytes, chunk);
            }
        }
    }
}
