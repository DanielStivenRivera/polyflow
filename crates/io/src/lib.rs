//! Import and export of meshes and images, plus the fast binary cache used
//! to reopen huge reference meshes quickly.
//!
//! Importers must treat every input file as untrusted: malformed data returns
//! an error, never panics.

#![forbid(unsafe_code)]
