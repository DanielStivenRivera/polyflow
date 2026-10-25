//! Binary cache files: named sections of plain data, written once and read
//! straight into typed arrays, with no parsing.
//!
//! A cache file stands for a source file: it records the source's path, size
//! and modification time, and the version of Polyflow that wrote it, and is
//! only used while all of them still match. Cache files are untrusted input
//! like any other (they can be truncated or corrupted): sections are checked
//! against the file's size, and callers validate what they read.
//!
//! Layout: the magic, the header's length (`u64`), a JSON header with the key
//! and the sections' offsets and lengths, then the sections, each aligned to
//! [`ALIGN`] bytes.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use bytemuck::Pod;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 8] = b"PFCACHE1";
/// Sections start at multiples of this, so they can be read in large,
/// aligned requests.
pub const ALIGN: u64 = 4096;
/// File extension of cache files.
pub const EXTENSION: &str = "pfcache";
/// Largest header accepted (it only holds a key and a section table).
const MAX_HEADER: u64 = 1 << 20;

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("invalid cache file: {0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> CacheError {
    CacheError::Invalid(message.into())
}

/// What a cache file stands for: a version of a source file, prepared by a
/// version of Polyflow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceKey {
    /// The source's absolute path.
    pub path: String,
    pub len: u64,
    /// Modification time, in nanoseconds since the Unix epoch.
    pub modified: u64,
    /// Version of the code that prepared the data; any change invalidates.
    pub version: String,
}

impl SourceKey {
    /// The key of `source` as it is now on disk.
    pub fn of(source: &Path, version: &str) -> std::io::Result<Self> {
        let path = std::fs::canonicalize(source)?;
        let metadata = std::fs::metadata(&path)?;
        let modified = metadata
            .modified()?
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        Ok(Self {
            path: path.to_string_lossy().into_owned(),
            len: metadata.len(),
            modified,
            version: version.to_owned(),
        })
    }

    /// File name of this source's cache: a hash of its path, so the same
    /// source always maps to the same cache file.
    pub fn file_name(&self) -> String {
        // FNV-1a: stable across runs and platforms, unlike std's hasher.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in self.path.bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
        }
        format!("{hash:016x}.{EXTENSION}")
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Header {
    key: SourceKey,
    sections: Vec<Section>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Section {
    name: String,
    offset: u64,
    length: u64,
}

/// Writes a cache file with the given sections, atomically: a reader sees
/// either the old file or the complete new one.
pub fn write(path: &Path, key: &SourceKey, sections: &[(&str, &[u8])]) -> std::io::Result<()> {
    let mut table = Vec::with_capacity(sections.len());
    // The header's size depends on the offsets, which depend on its size:
    // leave room for the largest header the table can need.
    let provisional = Header {
        key: key.clone(),
        sections: sections
            .iter()
            .map(|(name, _)| Section {
                name: (*name).to_owned(),
                offset: u64::MAX,
                length: u64::MAX,
            })
            .collect(),
    };
    let reserved = serde_json::to_vec(&provisional)
        .map_err(std::io::Error::other)?
        .len() as u64;
    let mut offset = align(MAGIC.len() as u64 + 8 + reserved);
    for (name, data) in sections {
        table.push(Section {
            name: (*name).to_owned(),
            offset,
            length: data.len() as u64,
        });
        offset = align(offset + data.len() as u64);
    }
    let header = serde_json::to_vec(&Header {
        key: key.clone(),
        sections: table.clone(),
    })
    .map_err(std::io::Error::other)?;

    // A name of its own, so two writers of the same cache cannot mix.
    let mut temp = path.as_os_str().to_owned();
    temp.push(format!(
        ".{}-{}.tmp",
        std::process::id(),
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    let temp = PathBuf::from(temp);
    let result = (|| {
        let mut file = std::io::BufWriter::with_capacity(1 << 20, File::create(&temp)?);
        file.write_all(MAGIC)?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        let mut at = MAGIC.len() as u64 + 8 + header.len() as u64;
        for (section, (_, data)) in table.iter().zip(sections) {
            pad(&mut file, section.offset - at)?;
            file.write_all(data)?;
            at = section.offset + data.len() as u64;
        }
        let file = file.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()
    })();
    match result.and_then(|()| std::fs::rename(&temp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&temp);
            Err(e)
        }
    }
}

fn align(offset: u64) -> u64 {
    offset.div_ceil(ALIGN) * ALIGN
}

fn pad(out: &mut impl Write, mut bytes: u64) -> std::io::Result<()> {
    const ZEROS: [u8; 4096] = [0; 4096];
    while bytes > 0 {
        let n = bytes.min(ZEROS.len() as u64) as usize;
        out.write_all(&ZEROS[..n])?;
        bytes -= n as u64;
    }
    Ok(())
}

/// An opened cache file whose key matched.
#[derive(Debug)]
pub struct CacheFile {
    path: PathBuf,
    sections: Vec<Section>,
    /// Size of the file.
    pub len: u64,
}

/// Opens the cache file at `path` if it exists and stands for `key`.
/// Returns `None` for a missing or stale cache, and an error for a damaged
/// one.
pub fn open(path: &Path, key: &SourceKey) -> Result<Option<CacheFile>, CacheError> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let len = file.metadata()?.len();
    let mut magic = [0u8; 8];
    let mut size = [0u8; 8];
    if file.read_exact(&mut magic).is_err() || &magic != MAGIC {
        return Err(invalid("not a Polyflow cache file (or an old one)"));
    }
    file.read_exact(&mut size)?;
    let size = u64::from_le_bytes(size);
    if size > MAX_HEADER.min(len) {
        return Err(invalid("the header is too large"));
    }
    let mut header = vec![0u8; size as usize];
    file.read_exact(&mut header)?;
    let header: Header =
        serde_json::from_slice(&header).map_err(|e| invalid(format!("header: {e}")))?;
    if header.key != *key {
        return Ok(None);
    }
    for section in &header.sections {
        let end = section.offset.checked_add(section.length);
        if end.is_none_or(|end| end > len) {
            return Err(invalid(format!(
                "section `{}` runs past the end",
                section.name
            )));
        }
    }
    Ok(Some(CacheFile {
        path: path.to_owned(),
        sections: header.sections,
        len,
    }))
}

impl CacheFile {
    /// Reads a section as an array of `T`, straight into its memory. Each
    /// call opens the file anew, so sections can be read in parallel.
    pub fn read<T: Pod>(&self, name: &str) -> Result<Vec<T>, CacheError> {
        let section = self
            .sections
            .iter()
            .find(|s| s.name == name)
            .ok_or_else(|| invalid(format!("no section `{name}`")))?;
        let size = size_of::<T>() as u64;
        if size == 0 || section.length % size != 0 {
            return Err(invalid(format!("section `{name}` has a partial element")));
        }
        let count = usize::try_from(section.length / size)
            .map_err(|_| invalid(format!("section `{name}` is too large")))?;
        // Zeroed memory comes straight from the OS, untouched until read into.
        let mut values: Vec<T> = bytemuck::allocation::zeroed_vec(count);
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(section.offset))?;
        file.read_exact(bytemuck::cast_slice_mut(&mut values))?;
        Ok(values)
    }
}

/// Deletes the least recently used cache files in `dir` until they take at
/// most `max_bytes`, keeping `keep`. Opening a cache counts as using it (see
/// [`touch`]).
pub fn prune(dir: &Path, max_bytes: u64, keep: &Path) -> std::io::Result<()> {
    let mut files: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;
        let cache = path.extension().is_some_and(|e| e == EXTENSION);
        // Leftover temporary files from writes that never finished.
        let stale_temp = path.extension().is_some_and(|e| e == "tmp")
            && metadata
                .modified()
                .ok()
                .and_then(|m| m.elapsed().ok())
                .is_some_and(|age| age.as_secs() > 24 * 3600);
        if stale_temp {
            let _ = std::fs::remove_file(&path);
        } else if cache && metadata.is_file() {
            let used = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            files.push((used, metadata.len(), path));
        }
    }
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort_by_key(|f| f.0);
    for (_, len, path) in files {
        if total <= max_bytes {
            break;
        }
        if path != keep && std::fs::remove_file(&path).is_ok() {
            total -= len;
        }
    }
    Ok(())
}

/// Marks a cache file as just used, for [`prune`].
pub fn touch(path: &Path) {
    if let Ok(file) = File::options().append(true).open(path) {
        let _ = file.set_modified(SystemTime::now());
    }
}

/// Total size of the cache files in `dir`.
pub fn size(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == EXTENSION))
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// Deletes every cache file in `dir`.
pub fn clear(dir: &Path) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == EXTENSION) {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(version: &str) -> SourceKey {
        SourceKey {
            path: "/sculpts/dragon.obj".into(),
            len: 1234,
            modified: 99,
            version: version.into(),
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("polyflow-cache-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn sections_round_trip() {
        let dir = temp_dir("round-trip");
        let path = dir.join(key("1").file_name());
        let floats: Vec<f32> = (0..10_000).map(|i| i as f32 * 0.5).collect();
        let words: Vec<u32> = vec![7, 8, 9];
        write(
            &path,
            &key("1"),
            &[
                ("floats", bytemuck::cast_slice(&floats)),
                ("empty", &[]),
                ("words", bytemuck::cast_slice(&words)),
            ],
        )
        .unwrap();
        let cache = open(&path, &key("1")).unwrap().unwrap();
        assert_eq!(cache.read::<f32>("floats").unwrap(), floats);
        assert_eq!(cache.read::<u32>("words").unwrap(), words);
        assert!(cache.read::<u32>("empty").unwrap().is_empty());
        assert!(cache.read::<u32>("missing").is_err());
        assert!(
            cache.read::<[u8; 8]>("words").is_err(),
            "12 bytes are not 8-byte elements"
        );
        // Another version (or source) does not use it.
        assert!(open(&path, &key("2")).unwrap().is_none());
        assert!(
            open(&dir.join("absent.pfcache"), &key("1"))
                .unwrap()
                .is_none()
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn damaged_files_are_errors() {
        let dir = temp_dir("damaged");
        let path = dir.join("a.pfcache");
        let data: Vec<u32> = (0..5000).collect();
        write(&path, &key("1"), &[("data", bytemuck::cast_slice(&data))]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        for len in [0, 4, 12, 40, 4096, bytes.len() - 1] {
            std::fs::write(&path, &bytes[..len]).unwrap();
            let result = open(&path, &key("1"));
            assert!(
                result.is_err()
                    || result
                        .unwrap()
                        .is_some_and(|c| c.read::<u32>("data").is_err()),
                "a truncation to {len} bytes went unnoticed"
            );
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn pruning_keeps_the_recent_ones() {
        let dir = temp_dir("prune");
        let data = vec![0u8; 10_000];
        let paths: Vec<PathBuf> = (0..3)
            .map(|i| dir.join(format!("{i}.{EXTENSION}")))
            .collect();
        for (i, path) in paths.iter().enumerate() {
            write(path, &key("1"), &[("data", &data)]).unwrap();
            let file = File::options().append(true).open(path).unwrap();
            let when =
                SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000 * (i as u64 + 1));
            file.set_modified(when).unwrap();
        }
        let one = std::fs::metadata(&paths[0]).unwrap().len();
        // Room for two: the oldest goes, unless it is the one to keep.
        prune(&dir, 2 * one, &paths[0]).unwrap();
        assert!(paths[0].exists() && !paths[1].exists() && paths[2].exists());
        assert_eq!(size(&dir), 2 * one);
        clear(&dir).unwrap();
        assert_eq!(size(&dir), 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
