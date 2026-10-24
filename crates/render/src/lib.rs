//! Vulkan 1.3 renderer built on `ash`.
//!
//! This is the only crate (besides the `polyflow` app) allowed to talk to the GPU.
//! Every `unsafe` block must carry a `// SAFETY:` comment.

pub mod bake;
pub mod context;
mod memory;
pub mod mesh;
mod overlay;
mod preview;
mod renderer;
mod swapchain;
mod timer;

pub use mesh::{CullStats, MeshData, MeshView};
pub use overlay::{Overlay, OverlayLine, OverlayPoint, OverlayStyle, OverlayVertex, Rgba8};
pub use preview::{PreviewData, PreviewMap, PreviewMode, PreviewVertex};
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
        "no GPU that can present to this window supports Vulkan 1.3 with dynamic rendering, \n         synchronization2, buffer device address and indirect draw count"
    )]
    NoSuitableGpu,
    #[error("{0}")]
    Other(String),
}
