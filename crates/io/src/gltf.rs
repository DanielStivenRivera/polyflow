//! glTF 2.0 import of reference meshes (`.gltf` with its buffers, or `.glb`)
//! and export of the low-poly as baked (`.glb`).
//!
//! The importer reads the triangles of every mesh in the default scene,
//! placed by their nodes' transforms. Buffers come from the GLB binary chunk,
//! `data:` URIs, or files next to the `.gltf`. Every offset and count is
//! checked against the data: malformed files return errors, never panic.
//! Compressed meshes (Draco, meshopt) are not supported.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use polyflow_core::glam::{Mat4, Quat, Vec3};
use serde_json::{Value, json};

use crate::Geometry;
use crate::lowpoly::MeshCorner;

const GLB_MAGIC: &[u8; 4] = b"glTF";
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;
/// Deepest node hierarchy followed (deeper ones are malformed or cyclic).
const MAX_NODE_DEPTH: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum GltfError {
    #[error("invalid GLB container: {0}")]
    Glb(&'static str),
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Invalid(String),
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("could not read buffer `{uri}`: {source}")]
    Buffer { uri: String, source: std::io::Error },
    #[error("the mesh is too large (more than 2^32 vertices or indices)")]
    TooLarge,
}

fn invalid(message: impl Into<String>) -> GltfError {
    GltfError::Invalid(message.into())
}

/// Parses a `.gltf` or `.glb` file. External buffers are read relative to
/// `base` (the file's folder); without it, only embedded data can be used.
pub fn parse(bytes: &[u8], base: Option<&Path>) -> Result<Geometry, GltfError> {
    let (document, bin) = split_glb(bytes)?;
    let root: Value = serde_json::from_slice(document)?;
    let required = root["extensionsRequired"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for extension in required.iter().filter_map(Value::as_str) {
        if extension != "KHR_mesh_quantization" {
            return Err(GltfError::Unsupported(format!(
                "the file needs the `{extension}` extension (export without compression)"
            )));
        }
    }
    let buffers = load_buffers(&root, bin, base)?;
    let file = File {
        root: &root,
        buffers,
    };

    // The meshes to read, with their world transforms.
    let mut instances: Vec<(usize, Mat4)> = Vec::new();
    let scene = root["scene"].as_u64().map(|s| s as usize).or_else(|| {
        root["scenes"]
            .as_array()
            .filter(|s| !s.is_empty())
            .map(|_| 0)
    });
    match scene {
        Some(s) => {
            let nodes = root["scenes"][s]["nodes"]
                .as_array()
                .ok_or_else(|| invalid(format!("scene {s} does not exist or has no nodes")))?;
            for n in nodes {
                file.collect_instances(index(n, "scene node")?, Mat4::IDENTITY, 0, &mut instances)?;
            }
        }
        // No scene: every mesh, untransformed.
        None => {
            let count = root["meshes"].as_array().map_or(0, Vec::len);
            instances.extend((0..count).map(|m| (m, Mat4::IDENTITY)));
        }
    }

    let mut geometry = Geometry::default();
    for (mesh, transform) in instances {
        file.read_mesh(mesh, transform, &mut geometry)?;
    }
    if geometry.positions.len() > u32::MAX as usize || geometry.indices.len() > u32::MAX as usize {
        return Err(GltfError::TooLarge);
    }
    Ok(geometry)
}

/// The JSON document and the binary chunk of a GLB, or the whole input as
/// JSON if it is not a GLB.
fn split_glb(bytes: &[u8]) -> Result<(&[u8], Option<&[u8]>), GltfError> {
    if !bytes.starts_with(GLB_MAGIC) {
        return Ok((bytes, None));
    }
    let u32_at = |at: usize| -> Result<u32, GltfError> {
        let b = bytes.get(at..at + 4).ok_or(GltfError::Glb("truncated"))?;
        Ok(u32::from_le_bytes(b.try_into().expect("4 bytes")))
    };
    if u32_at(4)? != 2 {
        return Err(GltfError::Glb("only version 2 is supported"));
    }
    let length = (u32_at(8)? as usize).min(bytes.len());
    let mut at = 12;
    let (mut json, mut bin) = (None, None);
    while at + 8 <= length {
        let chunk_length = u32_at(at)? as usize;
        let kind = u32_at(at + 4)?;
        let data = bytes
            .get(at + 8..at + 8 + chunk_length)
            .ok_or(GltfError::Glb("a chunk runs past the end of the file"))?;
        match kind {
            CHUNK_JSON if json.is_none() => json = Some(data),
            CHUNK_BIN if bin.is_none() => bin = Some(data),
            _ => {}
        }
        // Chunks are padded to 4 bytes.
        at += 8 + chunk_length.div_ceil(4) * 4;
    }
    Ok((json.ok_or(GltfError::Glb("no JSON chunk"))?, bin))
}

fn load_buffers(
    root: &Value,
    bin: Option<&[u8]>,
    base: Option<&Path>,
) -> Result<Vec<Vec<u8>>, GltfError> {
    let Some(buffers) = root["buffers"].as_array() else {
        return Ok(Vec::new());
    };
    buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| {
            let data = match buffer["uri"].as_str() {
                None if i == 0 => bin
                    .ok_or_else(|| invalid("buffer 0 has no URI and there is no GLB binary chunk"))?
                    .to_vec(),
                None => return Err(invalid(format!("buffer {i} has no URI"))),
                Some(uri) if uri.starts_with("data:") => decode_data_uri(uri)
                    .ok_or_else(|| invalid(format!("buffer {i} has an invalid data URI")))?,
                Some(uri) => {
                    let relative = percent_decode(uri);
                    let path = Path::new(&relative);
                    // Only files next to the glTF: no other schemes, no
                    // absolute paths.
                    if uri.contains("://") || path.is_absolute() || path.has_root() {
                        return Err(GltfError::Unsupported(format!("buffer URI `{uri}`")));
                    }
                    let base =
                        base.ok_or_else(|| invalid("external buffers need the file's folder"))?;
                    std::fs::read(base.join(path)).map_err(|source| GltfError::Buffer {
                        uri: uri.to_owned(),
                        source,
                    })?
                }
            };
            let length = buffer["byteLength"].as_u64().unwrap_or(data.len() as u64) as usize;
            if data.len() < length {
                return Err(invalid(format!(
                    "buffer {i} is shorter than its byteLength"
                )));
            }
            Ok(data)
        })
        .collect()
}

struct File<'a> {
    root: &'a Value,
    buffers: Vec<Vec<u8>>,
}

/// A view of accessor data: `count` elements of `components` values each.
struct Accessor<'a> {
    data: &'a [u8],
    stride: usize,
    count: usize,
    components: usize,
    component: u64,
    normalized: bool,
}

impl Accessor<'_> {
    fn component_size(component: u64) -> Option<usize> {
        match component {
            5120 | 5121 => Some(1),
            5122 | 5123 => Some(2),
            5125 | 5126 => Some(4),
            _ => None,
        }
    }

    /// Value `c` of element `i`, as a float (normalized if the accessor is).
    /// Bounds were checked when the accessor was made.
    fn get(&self, i: usize, c: usize) -> f32 {
        let size = Self::component_size(self.component).expect("checked");
        let at = i * self.stride + c * size;
        let b = &self.data[at..at + size];
        let n = self.normalized;
        match self.component {
            5120 => {
                let v = b[0] as i8 as f32;
                if n { (v / 127.0).max(-1.0) } else { v }
            }
            5121 => {
                let v = b[0] as f32;
                if n { v / 255.0 } else { v }
            }
            5122 => {
                let v = i16::from_le_bytes([b[0], b[1]]) as f32;
                if n { (v / 32767.0).max(-1.0) } else { v }
            }
            5123 => {
                let v = u16::from_le_bytes([b[0], b[1]]) as f32;
                if n { v / 65535.0 } else { v }
            }
            5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        }
    }

    /// Element `i` as an index.
    fn index(&self, i: usize) -> u32 {
        let size = Self::component_size(self.component).expect("checked");
        let at = i * self.stride;
        let b = &self.data[at..at + size];
        match self.component {
            5121 => b[0] as u32,
            5123 => u16::from_le_bytes([b[0], b[1]]) as u32,
            _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        }
    }
}

impl File<'_> {
    fn collect_instances(
        &self,
        node: usize,
        parent: Mat4,
        depth: usize,
        out: &mut Vec<(usize, Mat4)>,
    ) -> Result<(), GltfError> {
        if depth > MAX_NODE_DEPTH {
            return Err(invalid("the node hierarchy is too deep (or has a cycle)"));
        }
        let n = &self.root["nodes"][node];
        if !n.is_object() {
            return Err(invalid(format!("node {node} does not exist")));
        }
        let world = parent * local_transform(n)?;
        if let Some(mesh) = n.get("mesh") {
            out.push((index(mesh, "node mesh")?, world));
        }
        if let Some(children) = n["children"].as_array() {
            for child in children {
                self.collect_instances(index(child, "child node")?, world, depth + 1, out)?;
            }
        }
        Ok(())
    }

    fn accessor(&self, a: usize) -> Result<Accessor<'_>, GltfError> {
        let accessor = &self.root["accessors"][a];
        if !accessor.is_object() {
            return Err(invalid(format!("accessor {a} does not exist")));
        }
        if accessor.get("sparse").is_some() {
            return Err(GltfError::Unsupported(format!("sparse accessor {a}")));
        }
        let count = accessor["count"]
            .as_u64()
            .ok_or_else(|| invalid(format!("accessor {a} has no count")))?
            as usize;
        let component = accessor["componentType"].as_u64().unwrap_or(0);
        let size = Accessor::component_size(component)
            .ok_or_else(|| invalid(format!("accessor {a} has an unknown component type")))?;
        let components = match accessor["type"].as_str() {
            Some("SCALAR") => 1,
            Some("VEC2") => 2,
            Some("VEC3") => 3,
            Some("VEC4") => 4,
            _ => return Err(invalid(format!("accessor {a} has an unsupported type"))),
        };
        let element = size * components;
        let view = index(&accessor["bufferView"], "accessor buffer view")?;
        let view_json = &self.root["bufferViews"][view];
        let buffer = index(&view_json["buffer"], "buffer view buffer")?;
        let data = self
            .buffers
            .get(buffer)
            .ok_or_else(|| invalid(format!("buffer {buffer} does not exist")))?;
        let view_offset = view_json["byteOffset"].as_u64().unwrap_or(0) as usize;
        let view_length = view_json["byteLength"].as_u64().unwrap_or(0) as usize;
        let view_data = view_offset
            .checked_add(view_length)
            .and_then(|end| data.get(view_offset..end))
            .ok_or_else(|| invalid(format!("buffer view {view} runs past its buffer")))?;
        let stride = view_json["byteStride"]
            .as_u64()
            .map_or(element, |s| s as usize);
        if stride < element {
            return Err(invalid(format!(
                "buffer view {view} has a stride smaller than its elements"
            )));
        }
        let offset = accessor["byteOffset"].as_u64().unwrap_or(0) as usize;
        let needed = match count {
            0 => 0,
            n => (n - 1)
                .checked_mul(stride)
                .and_then(|x| x.checked_add(element))
                .ok_or_else(|| invalid(format!("accessor {a} is too large")))?,
        };
        let data = offset
            .checked_add(needed)
            .and_then(|end| view_data.get(offset..end))
            .ok_or_else(|| invalid(format!("accessor {a} runs past its buffer view")))?;
        Ok(Accessor {
            data,
            stride,
            count,
            components,
            component,
            normalized: accessor["normalized"].as_bool().unwrap_or(false),
        })
    }

    fn read_mesh(&self, mesh: usize, transform: Mat4, out: &mut Geometry) -> Result<(), GltfError> {
        let primitives = self.root["meshes"][mesh]["primitives"]
            .as_array()
            .ok_or_else(|| invalid(format!("mesh {mesh} does not exist or has no primitives")))?;
        // A mirroring transform turns triangles inside out: flip them back.
        let flip = transform.determinant() < 0.0;
        for primitive in primitives {
            let mode = primitive["mode"].as_u64().unwrap_or(4);
            if !matches!(mode, 4..=6) {
                continue; // points and lines
            }
            let Some(position) = primitive["attributes"].get("POSITION") else {
                continue;
            };
            let positions = self.accessor(index(position, "POSITION accessor")?)?;
            if positions.components != 3 {
                return Err(invalid("POSITION must be a VEC3 accessor"));
            }
            let base = out.positions.len();
            if base + positions.count > u32::MAX as usize {
                return Err(GltfError::TooLarge);
            }
            out.positions.extend((0..positions.count).map(|i| {
                let p = Vec3::new(
                    positions.get(i, 0),
                    positions.get(i, 1),
                    positions.get(i, 2),
                );
                transform.transform_point3(p)
            }));
            let indices: Vec<u32> = match primitive.get("indices") {
                Some(i) => {
                    let indices = self.accessor(index(i, "indices accessor")?)?;
                    if indices.components != 1 || !matches!(indices.component, 5121 | 5123 | 5125) {
                        return Err(invalid("indices must be unsigned integer scalars"));
                    }
                    (0..indices.count).map(|i| indices.index(i)).collect()
                }
                None => (0..positions.count as u32).collect(),
            };
            if let Some(&bad) = indices.iter().find(|&&i| i as usize >= positions.count) {
                return Err(invalid(format!(
                    "an index ({bad}) is past the {} vertices of its primitive",
                    positions.count
                )));
            }
            let mut push = |a: u32, b: u32, c: u32| {
                let [a, b, c] = [a, b, c].map(|i| i + base as u32);
                if flip {
                    out.indices.extend_from_slice(&[a, c, b]);
                } else {
                    out.indices.extend_from_slice(&[a, b, c]);
                }
            };
            match mode {
                4 => {
                    for &[a, b, c] in indices.as_chunks::<3>().0 {
                        push(a, b, c);
                    }
                }
                // Strips alternate winding; fans turn around the first vertex.
                5 => {
                    for k in 2..indices.len() {
                        let (a, b, c) = (indices[k - 2], indices[k - 1], indices[k]);
                        if k % 2 == 0 {
                            push(a, b, c)
                        } else {
                            push(b, a, c)
                        }
                    }
                }
                _ => {
                    for k in 2..indices.len() {
                        push(indices[0], indices[k - 1], indices[k]);
                    }
                }
            }
        }
        Ok(())
    }
}

/// A node's transform: its matrix, or its translation, rotation and scale.
fn local_transform(node: &Value) -> Result<Mat4, GltfError> {
    let floats = |key: &str, n: usize| -> Result<Option<Vec<f32>>, GltfError> {
        let Some(values) = node.get(key) else {
            return Ok(None);
        };
        let values: Option<Vec<f32>> = values
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Option<_>>()
            })
            .and_then(|v| v);
        match values {
            Some(v) if v.len() == n && v.iter().all(|x| x.is_finite()) => Ok(Some(v)),
            _ => Err(invalid(format!("invalid node {key}"))),
        }
    };
    if let Some(m) = floats("matrix", 16)? {
        return Ok(Mat4::from_cols_slice(&m));
    }
    let t = floats("translation", 3)?.map_or(Vec3::ZERO, |v| Vec3::new(v[0], v[1], v[2]));
    let r = floats("rotation", 4)?.map_or(Quat::IDENTITY, |v| {
        Quat::from_xyzw(v[0], v[1], v[2], v[3]).normalize()
    });
    let s = floats("scale", 3)?.map_or(Vec3::ONE, |v| Vec3::new(v[0], v[1], v[2]));
    Ok(Mat4::from_scale_rotation_translation(s, r, t))
}

fn index(value: &Value, what: &str) -> Result<usize, GltfError> {
    value
        .as_u64()
        .map(|i| i as usize)
        .ok_or_else(|| invalid(format!("missing or invalid {what}")))
}

fn decode_data_uri(uri: &str) -> Option<Vec<u8>> {
    let (header, data) = uri.split_once(',')?;
    if header.ends_with(";base64") {
        base64_decode(data)
    } else {
        Some(percent_decode(data).into_bytes())
    }
}

fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let value = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return None,
        } as u32)
    };
    let digits: Vec<u8> = text
        .bytes()
        .filter(|c| !c.is_ascii_whitespace() && *c != b'=')
        .collect();
    let mut out = Vec::with_capacity(digits.len() * 3 / 4);
    for group in digits.chunks(4) {
        let mut acc = 0u32;
        for &c in group {
            acc = acc << 6 | value(c)?;
        }
        match group.len() {
            4 => out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8, acc as u8]),
            3 => out.extend_from_slice(&[(acc >> 10) as u8, (acc >> 2) as u8]),
            2 => out.push((acc >> 4) as u8),
            _ => return None,
        }
    }
    Some(out)
}

fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        if bytes[i] == b'%'
            && let (Some(&h), Some(&l)) = (bytes.get(i + 1), bytes.get(i + 2))
            && let (Some(h), Some(l)) = (hex(h), hex(l))
        {
            out.push((h * 16 + l) as u8);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---- Export ------------------------------------------------------------------

/// Texture files (relative paths) to reference from the material.
#[derive(Debug, Clone, Default)]
pub struct BakedTextures {
    pub normal: Option<String>,
    pub occlusion: Option<String>,
}

impl BakedTextures {
    /// References the maps glTF can use: PNG files, by name, since they are
    /// written next to the GLB. Leave out normal maps that are not
    /// tangent-space with green up (OpenGL), which is what glTF expects.
    pub fn from_files(normal: Option<&Path>, occlusion: Option<&Path>) -> Self {
        let name = |path: Option<&Path>| {
            let path = path?;
            let png = path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("png"));
            png.then(|| path.file_name()?.to_str().map(str::to_owned))
                .flatten()
        };
        Self {
            normal: name(normal),
            occlusion: name(occlusion),
        }
    }
}

/// Writes triangles (three corners each) as a binary glTF: indexed, with
/// normals, tangents and UVs exactly as baked, and a material using the
/// baked maps. glTF's texture V runs down, so V is flipped; the tangent
/// frame on the surface is unchanged, which is what glTF's tangents and
/// OpenGL-style normal maps expect.
pub fn write_glb(
    corners: &[MeshCorner],
    textures: &BakedTextures,
    out: impl Write,
) -> std::io::Result<()> {
    // Share identical corners.
    let mut map: HashMap<[u32; 12], u32> = HashMap::new();
    let mut vertices: Vec<MeshCorner> = Vec::new();
    let indices: Vec<u32> = corners
        .iter()
        .map(|c| {
            let mut key = [0u32; 12];
            let values = (c.position.to_array().into_iter())
                .chain(c.normal.to_array())
                .chain(c.uv.to_array())
                .chain(c.tangent.to_array());
            for (k, v) in key.iter_mut().zip(values) {
                *k = v.to_bits();
            }
            *map.entry(key).or_insert_with(|| {
                vertices.push(*c);
                vertices.len() as u32 - 1
            })
        })
        .collect();

    let mut bin: Vec<u8> = Vec::new();
    let mut views = Vec::new();
    let mut add_view = |bin: &mut Vec<u8>, data: Vec<u8>, target: u32| {
        let offset = bin.len();
        bin.extend_from_slice(&data);
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        views.push(json!({
            "buffer": 0, "byteOffset": offset, "byteLength": data.len(), "target": target,
        }));
        views.len() - 1
    };
    let floats = |values: &mut dyn Iterator<Item = f32>| -> Vec<u8> {
        values.flat_map(f32::to_le_bytes).collect()
    };
    const ARRAY_BUFFER: u32 = 34962;
    const ELEMENT_ARRAY_BUFFER: u32 = 34963;
    let position = add_view(
        &mut bin,
        floats(&mut vertices.iter().flat_map(|v| v.position.to_array())),
        ARRAY_BUFFER,
    );
    let normal = add_view(
        &mut bin,
        floats(&mut vertices.iter().flat_map(|v| v.normal.to_array())),
        ARRAY_BUFFER,
    );
    let tangent = add_view(
        &mut bin,
        floats(&mut vertices.iter().flat_map(|v| v.tangent.to_array())),
        ARRAY_BUFFER,
    );
    let uv = add_view(
        &mut bin,
        floats(&mut vertices.iter().flat_map(|v| [v.uv.x, 1.0 - v.uv.y])),
        ARRAY_BUFFER,
    );
    let index_view = add_view(
        &mut bin,
        indices.iter().flat_map(|i| i.to_le_bytes()).collect(),
        ELEMENT_ARRAY_BUFFER,
    );
    let (min, max) = vertices.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), v| (lo.min(v.position), hi.max(v.position)),
    );
    let n = vertices.len();
    let mut accessors = vec![
        json!({"bufferView": position, "componentType": 5126, "count": n, "type": "VEC3",
               "min": min.to_array(), "max": max.to_array()}),
        json!({"bufferView": normal, "componentType": 5126, "count": n, "type": "VEC3"}),
        json!({"bufferView": tangent, "componentType": 5126, "count": n, "type": "VEC4"}),
        json!({"bufferView": uv, "componentType": 5126, "count": n, "type": "VEC2"}),
        json!({"bufferView": index_view, "componentType": 5125, "count": indices.len(), "type": "SCALAR"}),
    ];
    if n == 0 {
        // min and max need at least one vertex.
        accessors[0] =
            json!({"bufferView": position, "componentType": 5126, "count": 0, "type": "VEC3"});
    }

    let mut images = Vec::new();
    let mut material = json!({
        "name": "baked",
        "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.6},
    });
    for (key, uri) in [
        ("normalTexture", &textures.normal),
        ("occlusionTexture", &textures.occlusion),
    ] {
        if let Some(uri) = uri {
            images.push(json!({"uri": uri}));
            material[key] = json!({"index": images.len() - 1});
        }
    }
    let texture_list: Vec<Value> = (0..images.len()).map(|i| json!({"source": i})).collect();

    let mut document = json!({
        "asset": {"version": "2.0", "generator": format!("Polyflow {}", env!("CARGO_PKG_VERSION"))},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "lowpoly"}],
        "meshes": [{"name": "lowpoly", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TANGENT": 2, "TEXCOORD_0": 3},
            "indices": 4,
            "material": 0,
        }]}],
        "materials": [material],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": bin.len()}],
    });
    if !images.is_empty() {
        document["images"] = Value::Array(images);
        document["textures"] = Value::Array(texture_list);
    }
    let mut json = serde_json::to_vec(&document).map_err(std::io::Error::other)?;
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }

    let mut out = std::io::BufWriter::new(out);
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let total = u32::try_from(total)
        .map_err(|_| std::io::Error::other("the mesh is too large for a GLB"))?;
    out.write_all(GLB_MAGIC)?;
    out.write_all(&2u32.to_le_bytes())?;
    out.write_all(&total.to_le_bytes())?;
    out.write_all(&(json.len() as u32).to_le_bytes())?;
    out.write_all(&CHUNK_JSON.to_le_bytes())?;
    out.write_all(&json)?;
    out.write_all(&(bin.len() as u32).to_le_bytes())?;
    out.write_all(&CHUNK_BIN.to_le_bytes())?;
    out.write_all(&bin)?;
    out.flush()
}

/// [`write_glb`] to a file, atomically (see [`crate::retopo::save`]).
pub fn save_glb(
    path: &Path,
    corners: &[MeshCorner],
    textures: &BakedTextures,
) -> std::io::Result<()> {
    crate::save_atomically(path, |file| write_glb(corners, textures, file))
}

#[cfg(test)]
mod tests {
    use polyflow_core::glam::{Vec2, Vec4};

    use super::*;

    fn quad() -> Vec<MeshCorner> {
        let corner = |x: f32, y: f32| MeshCorner {
            position: Vec3::new(x, y, 0.0),
            normal: Vec3::Z,
            uv: Vec2::new(x, y),
            tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
        };
        vec![
            corner(0.0, 0.0),
            corner(1.0, 0.0),
            corner(1.0, 1.0),
            corner(0.0, 0.0),
            corner(1.0, 1.0),
            corner(0.0, 1.0),
        ]
    }

    #[test]
    fn exported_glb_reads_back() {
        let textures = BakedTextures {
            normal: Some("bake_normal.png".into()),
            occlusion: None,
        };
        let mut glb = Vec::new();
        write_glb(&quad(), &textures, &mut glb).unwrap();
        assert_eq!(glb.len() % 4, 0);
        let g = parse(&glb, None).unwrap();
        assert_eq!(g.positions.len(), 4, "shared corners are merged");
        assert_eq!(g.indices, [0, 1, 2, 0, 2, 3]);

        let (json, bin) = split_glb(&glb).unwrap();
        let root: Value = serde_json::from_slice(json).unwrap();
        assert_eq!(root["materials"][0]["normalTexture"]["index"], 0);
        assert_eq!(root["images"][0]["uri"], "bake_normal.png");
        // V is flipped: the corner at uv (0, 0) is at (0, 1) in glTF.
        let file = File {
            root: &root,
            buffers: vec![bin.unwrap().to_vec()],
        };
        let uv = file.accessor(3).unwrap();
        assert_eq!((uv.get(0, 0), uv.get(0, 1)), (0.0, 1.0));
        let tangent = file.accessor(2).unwrap();
        assert_eq!(tangent.get(0, 3), 1.0);
    }

    #[test]
    fn nodes_place_meshes() {
        // One triangle in a data URI, used by a node moved and mirrored.
        let positions: Vec<u8> = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let encoded = base64_encode(&positions);
        let document = json!({
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [
                {"translation": [10.0, 0.0, 0.0], "children": [1]},
                {"mesh": 0, "scale": [-1.0, 1.0, 1.0]},
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
            "bufferViews": [{"buffer": 0, "byteLength": 36}],
            "buffers": [{"byteLength": 36, "uri": format!("data:application/octet-stream;base64,{encoded}")}],
        });
        let g = parse(&serde_json::to_vec(&document).unwrap(), None).unwrap();
        assert_eq!(g.positions[1], Vec3::new(9.0, 0.0, 0.0));
        assert_eq!(g.indices, [0, 2, 1], "mirrored, so flipped back");
    }

    #[test]
    fn malformed_files_are_errors() {
        let mut glb = Vec::new();
        write_glb(&quad(), &BakedTextures::default(), &mut glb).unwrap();
        // Every truncation fails cleanly (or still parses).
        for len in 0..glb.len() {
            let _ = parse(&glb[..len], None);
        }
        // Corrupt bytes anywhere.
        for i in 0..glb.len() {
            let mut bad = glb.clone();
            bad[i] ^= 0xA5;
            let _ = parse(&bad, None);
        }
        let compressed = json!({"asset": {"version": "2.0"}, "extensionsRequired": ["KHR_draco_mesh_compression"]});
        assert!(matches!(
            parse(&serde_json::to_vec(&compressed).unwrap(), None),
            Err(GltfError::Unsupported(_))
        ));
        let escape = json!({"asset": {"version": "2.0"}, "buffers": [{"uri": "/etc/passwd", "byteLength": 1}]});
        assert!(parse(&serde_json::to_vec(&escape).unwrap(), Some(Path::new("."))).is_err());
        let cycle = json!({"asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"children": [0]}]});
        assert!(parse(&serde_json::to_vec(&cycle).unwrap(), None).is_err());
    }

    #[test]
    fn base64_round_trip() {
        for len in 0..10 {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(base64_decode(&base64_encode(&data)).unwrap(), data);
        }
        assert_eq!(percent_decode("my%20mesh.bin"), "my mesh.bin");
    }

    fn base64_encode(data: &[u8]) -> String {
        const DIGITS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
            for k in 0..=chunk.len() {
                out.push(DIGITS[(n >> (18 - 6 * k) & 63) as usize] as char);
            }
            for _ in chunk.len()..3 {
                out.push('=');
            }
        }
        out
    }
}
