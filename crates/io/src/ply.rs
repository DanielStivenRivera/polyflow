//! Stanford PLY import, ASCII and binary (either byte order).
//!
//! Only geometry is read: the `x`, `y`, `z` properties of the `vertex`
//! element and the index list of the `face` element (`vertex_indices` or
//! `vertex_index`). Polygons are fan-triangulated; other elements and
//! properties are skipped. Never panics on malformed input.

use polyflow_core::glam::Vec3;

use crate::Geometry;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PlyError {
    #[error("the file does not start with a PLY header")]
    NotPly,
    #[error("header line {line}: {message}")]
    Header { line: usize, message: String },
    #[error("the vertex element has no {0} property")]
    MissingCoordinate(char),
    #[error("the data ends before the {element} element is complete")]
    Truncated { element: String },
    #[error("{element} {index}: {message}")]
    Data {
        element: String,
        index: usize,
        message: String,
    },
    #[error("face references vertex {index}, but the file has {count} vertices")]
    IndexOutOfRange { index: i64, count: usize },
    #[error("the mesh is too large (more than 2^32 vertices or indices)")]
    TooLarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Ascii,
    LittleEndian,
    BigEndian,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "char" | "int8" => Scalar::I8,
            "uchar" | "uint8" => Scalar::U8,
            "short" | "int16" => Scalar::I16,
            "ushort" | "uint16" => Scalar::U16,
            "int" | "int32" => Scalar::I32,
            "uint" | "uint32" => Scalar::U32,
            "float" | "float32" => Scalar::F32,
            "double" | "float64" => Scalar::F64,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Scalar::I8 | Scalar::U8 => 1,
            Scalar::I16 | Scalar::U16 => 2,
            Scalar::I32 | Scalar::U32 | Scalar::F32 => 4,
            Scalar::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Property {
    Scalar(String, Scalar),
    /// A list: the count's type and the items' type.
    List(String, Scalar, Scalar),
}

impl Property {
    fn name(&self) -> &str {
        match self {
            Property::Scalar(name, _) | Property::List(name, ..) => name,
        }
    }
}

#[derive(Debug, Clone)]
struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

/// Parses PLY data.
pub fn parse(bytes: &[u8]) -> Result<Geometry, PlyError> {
    let (format, elements, body) = header(bytes)?;
    let mut reader = match format {
        Format::Ascii => Reader::Ascii(AsciiReader::new(body)),
        Format::LittleEndian => Reader::Binary(BinaryReader {
            data: body,
            at: 0,
            big: false,
        }),
        Format::BigEndian => Reader::Binary(BinaryReader {
            data: body,
            at: 0,
            big: true,
        }),
    };
    let mut geometry = Geometry::default();
    let mut vertex_count = None;
    let mut faces: Vec<u32> = Vec::new();
    let mut corners: Vec<u32> = Vec::new();
    for element in &elements {
        let truncated = || PlyError::Truncated {
            element: element.name.clone(),
        };
        // Reserve no more than the data could hold, whatever the header says.
        let max_items = body.len() / element_min_size(element, format).max(1);
        match element.name.as_str() {
            "vertex" if vertex_count.is_none() => {
                let coordinate = |axis: char| {
                    element
                        .properties
                        .iter()
                        .position(|p| matches!(p, Property::Scalar(name, _) if name.len() == 1 && name.starts_with(axis)))
                        .ok_or(PlyError::MissingCoordinate(axis))
                };
                let xyz = [coordinate('x')?, coordinate('y')?, coordinate('z')?];
                geometry.positions.reserve(element.count.min(max_items));
                let mut values = vec![0.0f64; element.properties.len()];
                for _ in 0..element.count {
                    for (value, property) in values.iter_mut().zip(&element.properties) {
                        *value = match property {
                            Property::Scalar(_, scalar) => {
                                reader.scalar(*scalar).ok_or_else(truncated)?
                            }
                            Property::List(_, count, item) => {
                                skip_list(&mut reader, *count, *item).ok_or_else(truncated)?;
                                0.0
                            }
                        };
                    }
                    let [x, y, z] = xyz.map(|i| values[i] as f32);
                    geometry.positions.push(Vec3::new(x, y, z));
                }
                if geometry.positions.len() > u32::MAX as usize {
                    return Err(PlyError::TooLarge);
                }
                vertex_count = Some(geometry.positions.len());
            }
            "face" => {
                let list = element
                    .properties
                    .iter()
                    .position(|p| {
                        matches!(p, Property::List(..))
                            && matches!(p.name(), "vertex_indices" | "vertex_index")
                    })
                    .or_else(|| {
                        element
                            .properties
                            .iter()
                            .position(|p| matches!(p, Property::List(..)))
                    });
                faces.reserve(element.count.min(max_items) * 3);
                for index in 0..element.count {
                    for (k, property) in element.properties.iter().enumerate() {
                        match property {
                            Property::List(_, count, item) if Some(k) == list => {
                                let n = reader.count(*count).ok_or_else(truncated)?;
                                corners.clear();
                                for _ in 0..n {
                                    let v = reader.scalar(*item).ok_or_else(truncated)?;
                                    if v < 0.0 || v.fract() != 0.0 || v > u32::MAX as f64 {
                                        return Err(PlyError::Data {
                                            element: element.name.clone(),
                                            index,
                                            message: format!("invalid vertex index {v}"),
                                        });
                                    }
                                    corners.push(v as u32);
                                }
                                // A separator, so polygons can be told apart.
                                faces.push(corners.len() as u32);
                                faces.extend_from_slice(&corners);
                            }
                            Property::List(_, count, item) => {
                                skip_list(&mut reader, *count, *item).ok_or_else(truncated)?;
                            }
                            Property::Scalar(_, scalar) => {
                                reader.scalar(*scalar).ok_or_else(truncated)?;
                            }
                        }
                    }
                }
            }
            _ => {
                for _ in 0..element.count {
                    for property in &element.properties {
                        match property {
                            Property::Scalar(_, scalar) => {
                                reader.scalar(*scalar).ok_or_else(truncated)?;
                            }
                            Property::List(_, count, item) => {
                                skip_list(&mut reader, *count, *item).ok_or_else(truncated)?;
                            }
                        }
                    }
                }
            }
        }
    }
    let count = geometry.positions.len();
    let mut k = 0;
    while k < faces.len() {
        let n = faces[k] as usize;
        let polygon = &faces[k + 1..k + 1 + n];
        if let Some(&bad) = polygon.iter().find(|&&i| i as usize >= count) {
            return Err(PlyError::IndexOutOfRange {
                index: bad as i64,
                count,
            });
        }
        geometry.push_polygon(polygon);
        k += n + 1;
    }
    if geometry.indices.len() > u32::MAX as usize {
        return Err(PlyError::TooLarge);
    }
    Ok(geometry)
}

/// Reads the header: the format, the elements, and the data after it.
fn header(bytes: &[u8]) -> Result<(Format, Vec<Element>, &[u8]), PlyError> {
    if !bytes.starts_with(b"ply") {
        return Err(PlyError::NotPly);
    }
    let mut format = None;
    let mut elements: Vec<Element> = Vec::new();
    let mut at = 0;
    let mut line_number = 0;
    loop {
        let Some(end) = bytes[at..].iter().position(|&b| b == b'\n') else {
            return Err(PlyError::Header {
                line: line_number + 1,
                message: "the header has no end_header line".into(),
            });
        };
        let line = &bytes[at..at + end];
        at += end + 1;
        line_number += 1;
        let line = String::from_utf8_lossy(line);
        let error = |message: &str| PlyError::Header {
            line: line_number,
            message: message.to_owned(),
        };
        let words: Vec<&str> = line.split_ascii_whitespace().collect();
        match words.as_slice() {
            ["ply"] if line_number == 1 => {}
            _ if line_number == 1 => return Err(PlyError::NotPly),
            ["format", kind, _version] => {
                format = Some(match *kind {
                    "ascii" => Format::Ascii,
                    "binary_little_endian" => Format::LittleEndian,
                    "binary_big_endian" => Format::BigEndian,
                    _ => return Err(error("unknown format")),
                });
            }
            ["comment", ..] | ["obj_info", ..] | [] => {}
            ["element", name, count] => {
                let count = count
                    .parse::<usize>()
                    .map_err(|_| error("invalid element count"))?;
                elements.push(Element {
                    name: (*name).to_owned(),
                    count,
                    properties: Vec::new(),
                });
            }
            ["property", "list", count, item, name] => {
                let (Some(count), Some(item)) = (Scalar::parse(count), Scalar::parse(item)) else {
                    return Err(error("unknown property type"));
                };
                if matches!(count, Scalar::F32 | Scalar::F64) {
                    return Err(error("a list count must be an integer"));
                }
                let element = elements
                    .last_mut()
                    .ok_or_else(|| error("property before any element"))?;
                element
                    .properties
                    .push(Property::List((*name).to_owned(), count, item));
            }
            ["property", kind, name] => {
                let kind = Scalar::parse(kind).ok_or_else(|| error("unknown property type"))?;
                let element = elements
                    .last_mut()
                    .ok_or_else(|| error("property before any element"))?;
                element
                    .properties
                    .push(Property::Scalar((*name).to_owned(), kind));
            }
            ["end_header"] => break,
            _ => return Err(error("unrecognized header line")),
        }
    }
    let format = format.ok_or(PlyError::Header {
        line: line_number,
        message: "no format line".into(),
    })?;
    Ok((format, elements, &bytes[at..]))
}

/// The smallest number of bytes one item of `element` can take.
fn element_min_size(element: &Element, format: Format) -> usize {
    element
        .properties
        .iter()
        .map(|p| match (format, p) {
            (Format::Ascii, _) => 2,
            (_, Property::Scalar(_, s)) => s.size(),
            (_, Property::List(_, count, _)) => count.size(),
        })
        .sum()
}

fn skip_list(reader: &mut Reader<'_>, count: Scalar, item: Scalar) -> Option<()> {
    let n = reader.count(count)?;
    for _ in 0..n {
        reader.scalar(item)?;
    }
    Some(())
}

enum Reader<'a> {
    Ascii(AsciiReader<'a>),
    Binary(BinaryReader<'a>),
}

impl Reader<'_> {
    fn scalar(&mut self, scalar: Scalar) -> Option<f64> {
        match self {
            Reader::Ascii(r) => r.number(),
            Reader::Binary(r) => r.scalar(scalar),
        }
    }

    /// A list's length.
    fn count(&mut self, scalar: Scalar) -> Option<usize> {
        let n = self.scalar(scalar)?;
        (n >= 0.0 && n.fract() == 0.0 && n <= u32::MAX as f64).then_some(n as usize)
    }
}

struct AsciiReader<'a> {
    words: std::str::SplitAsciiWhitespace<'a>,
}

impl<'a> AsciiReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        // Invalid UTF-8 cannot hold numbers: keep the valid prefix, so the
        // data simply ends there.
        let text = match std::str::from_utf8(data) {
            Ok(text) => text,
            Err(e) => std::str::from_utf8(&data[..e.valid_up_to()]).unwrap_or_default(),
        };
        Self {
            words: text.split_ascii_whitespace(),
        }
    }

    fn number(&mut self) -> Option<f64> {
        let word = self.words.next()?;
        fast_float2::parse(word).ok()
    }
}

struct BinaryReader<'a> {
    data: &'a [u8],
    at: usize,
    big: bool,
}

impl BinaryReader<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let bytes: [u8; N] = self.data.get(self.at..self.at + N)?.try_into().ok()?;
        self.at += N;
        Some(if self.big {
            let mut b = bytes;
            b.reverse();
            b
        } else {
            bytes
        })
    }

    fn scalar(&mut self, scalar: Scalar) -> Option<f64> {
        Some(match scalar {
            Scalar::I8 => i8::from_le_bytes(self.take()?) as f64,
            Scalar::U8 => u8::from_le_bytes(self.take()?) as f64,
            Scalar::I16 => i16::from_le_bytes(self.take()?) as f64,
            Scalar::U16 => u16::from_le_bytes(self.take()?) as f64,
            Scalar::I32 => i32::from_le_bytes(self.take()?) as f64,
            Scalar::U32 => u32::from_le_bytes(self.take()?) as f64,
            Scalar::F32 => f32::from_le_bytes(self.take()?) as f64,
            Scalar::F64 => f64::from_le_bytes(self.take()?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = "ply\nformat {format} 1.0\ncomment made by hand\n\
        element vertex 4\nproperty float x\nproperty float y\nproperty float z\n\
        property uchar red\n\
        element face 2\nproperty list uchar int vertex_indices\n\
        element edge 1\nproperty int vertex1\nproperty int vertex2\nend_header\n";

    fn header(format: &str) -> Vec<u8> {
        HEADER.replace("{format}", format).into_bytes()
    }

    fn binary(big: bool) -> Vec<u8> {
        let mut data = header(if big {
            "binary_big_endian"
        } else {
            "binary_little_endian"
        });
        let f = |data: &mut Vec<u8>, x: f32| {
            data.extend(if big {
                x.to_be_bytes()
            } else {
                x.to_le_bytes()
            })
        };
        let i = |data: &mut Vec<u8>, x: i32| {
            data.extend(if big {
                x.to_be_bytes()
            } else {
                x.to_le_bytes()
            })
        };
        for (x, y) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            f(&mut data, x);
            f(&mut data, y);
            f(&mut data, 0.5);
            data.push(255);
        }
        data.push(4);
        for k in [0, 1, 2, 3] {
            i(&mut data, k);
        }
        data.push(2);
        i(&mut data, 0);
        i(&mut data, 1);
        i(&mut data, 0);
        i(&mut data, 2);
        data
    }

    #[test]
    fn reads_every_format() {
        let mut ascii = header("ascii");
        ascii.extend_from_slice(
            b"0 0 0.5 255\n1 0 0.5 255\n1 1 0.5 255\n0 1 0.5 255\n4 0 1 2 3\n2 0 1\n0 2\n",
        );
        for data in [ascii, binary(false), binary(true)] {
            let g = parse(&data).unwrap();
            assert_eq!(g.positions.len(), 4);
            assert_eq!(g.positions[2], Vec3::new(1.0, 1.0, 0.5));
            assert_eq!(g.indices, [0, 1, 2, 0, 2, 3]);
            assert_eq!(g.skipped_faces, 1, "the two-vertex face");
        }
    }

    #[test]
    fn malformed_files_are_errors() {
        assert_eq!(parse(b"solid stl"), Err(PlyError::NotPly));
        assert!(matches!(
            parse(b"ply\nformat ascii 1.0\n"),
            Err(PlyError::Header { .. })
        ));
        let mut data = binary(false);
        data.truncate(data.len() - 3);
        assert!(matches!(parse(&data), Err(PlyError::Truncated { .. })));
        let mut ascii = header("ascii");
        ascii.extend_from_slice(b"0 0 0 1\n1 0 0 1\n1 1 0 1\n0 1 0 1\n3 0 1 9\n0\n0 0\n");
        assert!(matches!(
            parse(&ascii),
            Err(PlyError::IndexOutOfRange { index: 9, .. })
        ));
        // A huge count in the header must not allocate or loop forever.
        let huge = b"ply\nformat binary_little_endian 1.0\nelement vertex 99999999999\n\
            property float x\nproperty float y\nproperty float z\nend_header\n\0\0\0\0";
        assert!(matches!(parse(huge), Err(PlyError::Truncated { .. })));
        let no_z = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n\
            property float y\nend_header\n0 0\n";
        assert_eq!(parse(no_z), Err(PlyError::MissingCoordinate('z')));
        // Every prefix of a valid file fails cleanly.
        let data = binary(true);
        for len in 0..data.len() {
            let _ = parse(&data[..len]);
        }
    }
}
