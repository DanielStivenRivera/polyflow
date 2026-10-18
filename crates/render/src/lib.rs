//! Vulkan 1.3 renderer built on `ash`.
//!
//! This is the only crate (besides `polyflow-app`) allowed to talk to the GPU.
//! Every `unsafe` block must carry a `// SAFETY:` comment.
