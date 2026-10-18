//! Texture baking: projects detail from the reference mesh onto the
//! retopology mesh (tangent-space normals, AO, height, cavity, curvature).
//!
//! Contains a CPU reference implementation used to validate the GPU path.

#![forbid(unsafe_code)]
