//! Core data structures for Polyflow.
//!
//! This crate has no GPU or windowing dependencies and must stay testable on a
//! headless machine. It will host the half-edge retopology mesh and the
//! command / undo system.

#![forbid(unsafe_code)]

pub use glam;
