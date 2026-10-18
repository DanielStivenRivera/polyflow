//! Vulkan 1.3 renderer built on `ash`.
//!
//! This is the only crate (besides the `polyflow` app) allowed to talk to the GPU.
//! Every `unsafe` block must carry a `// SAFETY:` comment.

mod context;
mod memory;
mod mesh;
mod renderer;
mod swapchain;

pub use renderer::{FRAMES_IN_FLIGHT, FrameInput, Renderer, UiFrame, Viewport};

/// Errors produced by the renderer.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("failed to load the Vulkan library: {0}")]
    Loading(#[from] ash::LoadingError),
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] ash::vk::Result),
    #[error("GPU memory allocation failed: {0}")]
    Allocation(#[from] gpu_allocator::AllocationError),
    #[error("UI renderer error: {0}")]
    Ui(#[from] egui_ash_renderer::RendererError),
    #[error("window handle error: {0}")]
    Window(String),
    #[error(
        "no GPU with Vulkan 1.3, dynamic rendering and synchronization2 that can present to this window"
    )]
    NoSuitableGpu,
    #[error("{0}")]
    Other(String),
}
