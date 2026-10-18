//! Frame orchestration: swapchain, frames in flight, scene and UI passes.

use std::sync::Arc;

use ash::vk;
use egui_ash_renderer::allocator::GpuAllocator;
use egui_ash_renderer::{DynamicRendering, Options, RenderMode};
use polyflow_core::glam::Vec3;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::RenderError;
use crate::context::GpuContext;
use crate::memory::{GpuImage, subresource_range};
use crate::mesh::{CullStats, GpuMesh, MeshData, MeshPipeline, MeshView};
use crate::swapchain::Swapchain;
use crate::timer::GpuTimer;

/// Number of frames the CPU may record ahead of the GPU.
pub const FRAMES_IN_FLIGHT: usize = 2;

/// Everything needed to draw one frame.
#[derive(Debug)]
pub struct FrameInput<'a> {
    /// Region of the window the 3D scene is drawn into, in physical pixels.
    pub viewport: Viewport,
    /// Camera and display settings for the reference mesh.
    pub view: MeshView,
    /// Linear RGB.
    pub clear_color: Vec3,
    pub ui: Option<UiFrame<'a>>,
}

/// A rectangle in physical pixels, origin at the top-left of the window.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Viewport {
    /// Integer pixel rectangle clamped to `extent`; `None` if it is empty.
    fn clamped_rect(&self, extent: vk::Extent2D) -> Option<vk::Rect2D> {
        let x0 = self.x.round().clamp(0.0, extent.width as f32) as i32;
        let y0 = self.y.round().clamp(0.0, extent.height as f32) as i32;
        let x1 = (self.x + self.width)
            .round()
            .clamp(0.0, extent.width as f32) as i32;
        let y1 = (self.y + self.height)
            .round()
            .clamp(0.0, extent.height as f32) as i32;
        (x1 > x0 && y1 > y0).then(|| vk::Rect2D {
            offset: vk::Offset2D { x: x0, y: y0 },
            extent: vk::Extent2D {
                width: (x1 - x0) as u32,
                height: (y1 - y0) as u32,
            },
        })
    }
}

/// Output of an egui pass, ready to be rendered.
#[derive(Debug)]
pub struct UiFrame<'a> {
    pub primitives: &'a [egui::ClippedPrimitive],
    /// Consumed by [`Renderer::render`], which leaves it empty.
    pub textures: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

struct Frame {
    cmd: vk::CommandBuffer,
    image_available: vk::Semaphore,
    in_flight: vk::Fence,
    /// egui textures to free once this frame's fence proves the GPU is done.
    textures_to_free: Vec<egui::TextureId>,
}

/// The Vulkan renderer for one window.
pub struct Renderer {
    swapchain: Swapchain,
    depth: GpuImage,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame_index: usize,
    mesh_pipeline: MeshPipeline,
    mesh: Option<GpuMesh>,
    timer: GpuTimer,
    /// GPU time of the most recently completed frame.
    gpu_time: Option<std::time::Duration>,
    /// Culling results of the most recently completed frame.
    cull_stats: Option<CullStats>,
    egui: Option<egui_ash_renderer::Renderer<GpuAllocator>>,
    window_extent: vk::Extent2D,
    vsync: bool,
    needs_recreate: bool,
    // Declared last: dropped after everything above has been destroyed.
    ctx: GpuContext,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("ctx", &self.ctx)
            .field("swapchain", &self.swapchain)
            .finish_non_exhaustive()
    }
}

impl Renderer {
    pub fn new(
        window: &(impl HasDisplayHandle + HasWindowHandle),
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<Self, RenderError> {
        let ctx = GpuContext::new(window)?;
        let window_extent = vk::Extent2D { width, height };
        let device = &ctx.device;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid device and queue family.
        let command_pool = unsafe { device.create_command_pool(&pool_info, None)? };
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        // SAFETY: the pool belongs to this device.
        let cmds = unsafe { device.allocate_command_buffers(&alloc_info)? };
        let frames = cmds
            .into_iter()
            .map(|cmd| {
                // SAFETY: plain object creation; destroyed in `Drop`.
                unsafe {
                    Ok(Frame {
                        cmd,
                        image_available: device.create_semaphore(&Default::default(), None)?,
                        in_flight: device.create_fence(
                            &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                            None,
                        )?,
                        textures_to_free: vec![],
                    })
                }
            })
            .collect::<Result<Vec<_>, vk::Result>>()?;

        let swapchain = Swapchain::new(&ctx, window_extent, vsync, vk::SwapchainKHR::null())?;
        if !is_srgb(swapchain.format) {
            log::warn!(
                "No sRGB swapchain format available ({:?}); colors will be off",
                swapchain.format
            );
        }
        let depth_format = pick_depth_format(&ctx)?;
        let depth = create_depth(&ctx, swapchain.extent, depth_format)?;
        let mesh_pipeline = MeshPipeline::new(&ctx, swapchain.format, depth_format)?;
        let timer = GpuTimer::new(&ctx)?;
        let egui = egui_ash_renderer::Renderer::with_gpu_allocator(
            Arc::clone(&ctx.allocator),
            ctx.device.clone(),
            RenderMode::DynamicRendering(DynamicRendering {
                color_attachment_format: swapchain.format,
                depth_attachment_format: Some(depth_format),
                stencil_attachment_format: None,
            }),
            Options {
                in_flight_frames: FRAMES_IN_FLIGHT,
                enable_depth_test: false,
                enable_depth_write: false,
                srgb_framebuffer: is_srgb(swapchain.format),
            },
        )?;

        Ok(Self {
            swapchain,
            depth,
            command_pool,
            frames,
            frame_index: 0,
            mesh_pipeline,
            mesh: None,
            timer,
            gpu_time: None,
            cull_stats: None,
            egui: Some(egui),
            window_extent,
            vsync,
            needs_recreate: false,
            ctx,
        })
    }

    pub fn device_name(&self) -> &str {
        self.ctx.device_name()
    }

    /// Meshlet culling results of the most recently completed frame.
    pub fn cull_stats(&self) -> Option<CullStats> {
        self.cull_stats
    }

    /// GPU execution time of the most recently completed frame.
    pub fn gpu_frame_time(&self) -> Option<std::time::Duration> {
        self.gpu_time
    }

    pub fn max_texture_side(&self) -> usize {
        self.ctx.properties.limits.max_image_dimension2_d as usize
    }

    pub fn present_mode_name(&self) -> &'static str {
        match self.swapchain.present_mode {
            vk::PresentModeKHR::FIFO => "FIFO (vsync)",
            vk::PresentModeKHR::MAILBOX => "Mailbox",
            vk::PresentModeKHR::IMMEDIATE => "Immediate",
            vk::PresentModeKHR::FIFO_RELAXED => "FIFO relaxed",
            _ => "other",
        }
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        if vsync != self.vsync {
            self.vsync = vsync;
            self.needs_recreate = true;
        }
    }

    /// Call when the window's framebuffer size changes.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.window_extent = vk::Extent2D { width, height };
        self.needs_recreate = true;
    }

    /// Replaces the displayed mesh. Blocks until the upload has finished.
    pub fn set_mesh(&mut self, mesh: &MeshData) -> Result<(), RenderError> {
        // SAFETY: waiting for idle so the old mesh is no longer in use.
        unsafe { self.ctx.device.device_wait_idle()? };
        if let Some(mut old) = self.mesh.take() {
            old.destroy(&self.ctx);
        }
        self.mesh = Some(GpuMesh::upload(&self.ctx, self.command_pool, mesh)?);
        Ok(())
    }

    /// Records, submits and presents one frame.
    pub fn render(&mut self, mut input: FrameInput<'_>) -> Result<(), RenderError> {
        // Take the texture changes out first: egui panics if a delta is dropped
        // unapplied, which would otherwise happen on any early error return.
        let (textures_to_set, textures_to_free) = input
            .ui
            .as_mut()
            .map(|ui| {
                (
                    std::mem::take(&mut ui.textures.set),
                    std::mem::take(&mut ui.textures.free),
                )
            })
            .unwrap_or_default();

        let device = self.ctx.device.clone();
        let frame = &mut self.frames[self.frame_index];
        // SAFETY: the fence belongs to this device.
        unsafe { device.wait_for_fences(&[frame.in_flight], true, u64::MAX)? };
        if let Some(time) = self.timer.collect(&device, self.frame_index) {
            self.gpu_time = Some(time);
        }
        if let Some(stats) = self
            .mesh
            .as_mut()
            .and_then(|m| m.take_stats(self.frame_index))
        {
            self.cull_stats = Some(stats);
        }

        let egui = self.egui.as_mut().expect("egui renderer lives until drop");
        for id in frame.textures_to_free.drain(..) {
            egui.free_texture(id)?;
        }
        // Applied even on frames that end up not being presented, since egui
        // considers the delta handled.
        if !textures_to_set.is_empty() {
            // Rare (font atlas changes): wait so no in-flight frame reads a texture being rewritten.
            // SAFETY: plain wait on a valid device.
            unsafe { device.device_wait_idle()? };
            for (id, deltas) in &textures_to_set {
                // Order matters within one texture: a whole-image delta comes first.
                for delta in deltas {
                    egui.set_texture(self.ctx.queue, self.command_pool, *id, delta)?;
                }
            }
        }
        frame.textures_to_free.extend(textures_to_free);

        if self.window_extent.width == 0 || self.window_extent.height == 0 {
            return Ok(()); // Minimized.
        }
        if self.needs_recreate && !self.recreate_swapchain()? {
            return Ok(()); // The surface has no area right now (e.g. minimized).
        }

        let frame = &self.frames[self.frame_index];
        let image_index = match self.swapchain.acquire(frame.image_available) {
            Ok((index, suboptimal)) => {
                self.needs_recreate |= suboptimal;
                index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_recreate = true;
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };
        // Only reset once we know work will be submitted, or the next wait deadlocks.
        // SAFETY: the fence is signaled and not in use.
        unsafe { device.reset_fences(&[frame.in_flight])? };

        self.record(frame.cmd, image_index, &input)?;

        let frame = &self.frames[self.frame_index];
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(frame.image_available)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let signals = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.swapchain.render_finished[image_index as usize])
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(frame.cmd)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .signal_semaphore_infos(&signals)
            .command_buffer_infos(&cmds);
        // SAFETY: the command buffer is fully recorded and every referenced
        // resource outlives the frame's fence.
        unsafe { device.queue_submit2(self.ctx.queue, &[submit], frame.in_flight)? };

        match self.swapchain.present(self.ctx.queue, image_index) {
            Ok(suboptimal) => self.needs_recreate |= suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.needs_recreate = true,
            Err(e) => return Err(e.into()),
        }
        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }

    fn record(
        &mut self,
        cmd: vk::CommandBuffer,
        image_index: u32,
        input: &FrameInput<'_>,
    ) -> Result<(), RenderError> {
        let device = &self.ctx.device;
        let image = self.swapchain.images[image_index as usize];
        let view = self.swapchain.views[image_index as usize];
        let extent = self.swapchain.extent;

        // SAFETY: the frame's fence was waited on, so `cmd` is not in use; all
        // recorded resources stay alive until the fence signals again.
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.timer.begin(device, cmd, self.frame_index);
            let mesh_visible = input.viewport.clamped_rect(extent).is_some();
            if let Some(mesh) = self.mesh.as_mut().filter(|_| mesh_visible) {
                self.mesh_pipeline
                    .record_cull(&self.ctx, cmd, mesh, self.frame_index, &input.view);
            }

            let to_attachment = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(image)
                    .subresource_range(subresource_range(vk::ImageAspectFlags::COLOR)),
                // The depth image is shared by all frames in flight; this also
                // orders this frame's depth writes after the previous frame's.
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .dst_access_mask(
                        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    )
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .image(self.depth.image)
                    .subresource_range(subresource_range(vk::ImageAspectFlags::DEPTH)),
            ];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_attachment),
            );

            let c = input.clear_color;
            let color_attachments = [vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [c.x, c.y, c.z, 1.0],
                    },
                })];
            // Reverse-Z: clear to the far plane at 0.
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.depth.view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                });
            let rendering = vk::RenderingInfo::default()
                .render_area(extent.into())
                .layer_count(1)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment);
            device.cmd_begin_rendering(cmd, &rendering);

            if let (Some(mesh), Some(rect)) = (&self.mesh, input.viewport.clamped_rect(extent)) {
                let viewport = vk::Viewport {
                    x: rect.offset.x as f32,
                    y: rect.offset.y as f32,
                    width: rect.extent.width as f32,
                    height: rect.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cmd, 0, &[viewport]);
                device.cmd_set_scissor(cmd, 0, &[rect]);
                self.mesh_pipeline
                    .draw(device, cmd, mesh, self.frame_index, &input.view);
            }

            // The UI renderer sets its own viewport and scissors.
            if let Some(ui) = &input.ui {
                let egui = self.egui.as_mut().expect("egui renderer lives until drop");
                egui.cmd_draw(cmd, extent, ui.pixels_per_point, ui.primitives)?;
            }

            device.cmd_end_rendering(cmd);

            let to_present = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(image)
                .subresource_range(subresource_range(vk::ImageAspectFlags::COLOR))];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_present),
            );
            self.timer.end(device, cmd, self.frame_index);
            device.end_command_buffer(cmd)?;
        }
        Ok(())
    }

    /// Returns false, leaving the old swapchain in place, when the surface
    /// currently has zero size.
    fn recreate_swapchain(&mut self) -> Result<bool, RenderError> {
        let extent = Swapchain::current_surface_extent(&self.ctx, self.window_extent)?;
        if extent.width == 0 || extent.height == 0 {
            return Ok(false);
        }
        // SAFETY: waiting for idle so no swapchain or depth resource is in use.
        unsafe { self.ctx.device.device_wait_idle()? };
        let mut new = Swapchain::new(
            &self.ctx,
            self.window_extent,
            self.vsync,
            self.swapchain.handle,
        )?;
        if new.format != self.swapchain.format {
            new.destroy(&self.ctx);
            return Err(RenderError::Other(
                "swapchain format changed; pipelines would need rebuilding".into(),
            ));
        }
        let mut old = std::mem::replace(&mut self.swapchain, new);
        old.destroy(&self.ctx);

        let depth = create_depth(&self.ctx, self.swapchain.extent, self.depth.format)?;
        let mut old_depth = std::mem::replace(&mut self.depth, depth);
        old_depth.destroy(&self.ctx);
        self.needs_recreate = false;
        Ok(true)
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // SAFETY: after waiting for idle nothing is in use; objects are destroyed
        // before the context (dropped after this body).
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
            // Drops its pipeline, buffers and its clone of the allocator handle.
            self.egui = None;
            if let Some(mut mesh) = self.mesh.take() {
                mesh.destroy(&self.ctx);
            }
            self.mesh_pipeline.destroy(&self.ctx);
            self.timer.destroy(&self.ctx);
            self.depth.destroy(&self.ctx);
            self.swapchain.destroy(&self.ctx);
            for frame in &self.frames {
                self.ctx
                    .device
                    .destroy_semaphore(frame.image_available, None);
                self.ctx.device.destroy_fence(frame.in_flight, None);
            }
            self.ctx
                .device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

fn is_srgb(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}

fn pick_depth_format(ctx: &GpuContext) -> Result<vk::Format, RenderError> {
    // Float depth pairs with reverse-Z for near-uniform precision.
    [vk::Format::D32_SFLOAT, vk::Format::X8_D24_UNORM_PACK32]
        .into_iter()
        .find(|&format| {
            // SAFETY: plain query.
            let props = unsafe {
                ctx.instance
                    .get_physical_device_format_properties(ctx.physical_device, format)
            };
            props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        })
        .ok_or(RenderError::Other("no supported depth format".into()))
}

fn create_depth(
    ctx: &GpuContext,
    extent: vk::Extent2D,
    format: vk::Format,
) -> Result<GpuImage, RenderError> {
    GpuImage::new(
        ctx,
        extent,
        format,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::ImageAspectFlags::DEPTH,
        "depth",
    )
}
