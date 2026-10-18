//! Core data structures for Polyflow.
//!
//! This crate has no GPU or windowing dependencies and must stay testable on a
//! headless machine. It hosts the half-edge retopology mesh and its undo
//! history.

#![forbid(unsafe_code)]

pub mod bounds;
pub mod camera;
pub mod halfedge;
pub mod history;
pub mod mesh;
pub mod primitives;

pub use glam;
