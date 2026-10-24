//! Overlay geometry drawn on top of the reference mesh: the retopology mesh
//! and tool previews. Data comes from the CPU every frame it changes.

use std::io::Cursor;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use polyflow_core::glam::{Mat4, Vec2, Vec3, Vec4};

use crate::RenderError;
use crate::context::GpuContext;
use crate::memory::GpuBuffer;
use crate::renderer::FRAMES_IN_FLIGHT;

const OVERLAY_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/overlay.spv"));

/// An sRGB color with alpha, 8 bits per channel.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq, Eq)]
pub struct Rgba8(pub [u8; 4]);

impl Rgba8 {
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self([r, g, b, a])
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct OverlayVertex {
    pub position: Vec3,
    pub color: Rgba8,
    pub uv: Vec2,
    /// 1 to shade the triangle with a checker pattern from `uv` (tinted by
    /// `color`'s alpha), 0 for a flat `color`.
    pub checker: f32,
}

impl OverlayVertex {
    /// A vertex of a flat-colored triangle.
    pub fn flat(position: Vec3, color: Rgba8) -> Self {
        Self {
            position,
            color,
            uv: Vec2::ZERO,
            checker: 0.0,
        }
    }

    /// A vertex of a triangle shaded with a UV checker.
    pub fn checker(position: Vec3, color: Rgba8, uv: Vec2) -> Self {
        Self {
            position,
            color,
            uv,
            checker: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct OverlayLine {
    pub a: Vec3,
    pub b: Vec3,
    pub color: Rgba8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct OverlayPoint {
    pub position: Vec3,
    pub color: Rgba8,
    /// Diameter in pixels.
    pub size: f32,
}

/// A batch of overlay geometry in world space.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Overlay {
    /// Three vertices per filled triangle (alpha-blended).
    pub triangles: Vec<OverlayVertex>,
    pub lines: Vec<OverlayLine>,
    pub points: Vec<OverlayPoint>,
}

impl Overlay {
    pub fn clear(&mut self) {
        self.triangles.clear();
        self.lines.clear();
        self.points.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty() && self.lines.is_empty() && self.points.is_empty()
    }
}

/// How overlays are drawn this frame.
#[derive(Debug, Clone, Copy)]
pub struct OverlayStyle {
    pub line_width: f32,
    /// Draw through the reference mesh instead of being hidden by it.
    pub x_ray: bool,
    /// Fraction of its distance to the eye by which overlay geometry is
    /// pulled forward. Flat faces and edges between snapped vertices sink
    /// under convex surfaces; larger values keep them visible, at the cost
    /// of showing through thin parts of the model.
    pub surface_offset: f32,
}

impl Default for OverlayStyle {
    fn default() -> Self {
        Self {
            line_width: 1.5,
            x_ray: false,
            surface_offset: 0.01,
        }
    }
}

/// Must match `PushConstants` in `shaders/overlay.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OverlayPushConstants {
    view_proj: Mat4,
    eye: Vec4,
    viewport_size: Vec2,
    line_width: f32,
    depth_pull: f32,
}

/// A host-visible buffer per frame in flight, grown on demand.
#[derive(Debug)]
struct StreamBuffer {
    buffers: Vec<GpuBuffer>,
    capacity: Vec<u64>,
    usage: vk::BufferUsageFlags,
    name: &'static str,
}

impl StreamBuffer {
    fn new(usage: vk::BufferUsageFlags, name: &'static str) -> Self {
        Self {
            buffers: (0..FRAMES_IN_FLIGHT).map(|_| GpuBuffer::null()).collect(),
            capacity: vec![0; FRAMES_IN_FLIGHT],
            usage,
            name,
        }
    }

    /// Copies `parts` back to back into the buffer of `frame`. The caller
    /// guarantees the GPU no longer reads that frame's buffer.
    fn write(
        &mut self,
        ctx: &GpuContext,
        frame: usize,
        parts: &[&[u8]],
    ) -> Result<(), RenderError> {
        let size: u64 = parts.iter().map(|p| p.len() as u64).sum();
        if size == 0 {
            return Ok(());
        }
        if size > self.capacity[frame] {
            let capacity = size.next_power_of_two().max(64 << 10);
            self.buffers[frame].destroy(ctx);
            self.buffers[frame] = GpuBuffer::new(
                ctx,
                capacity,
                self.usage,
                MemoryLocation::CpuToGpu,
                self.name,
            )?;
            self.capacity[frame] = capacity;
        }
        let mapped = self.buffers[frame].mapped_mut().ok_or(RenderError::Other(
            "overlay buffer is not host visible".into(),
        ))?;
        let mut offset = 0;
        for part in parts {
            mapped[offset..offset + part.len()].copy_from_slice(part);
            offset += part.len();
        }
        Ok(())
    }

    fn buffer(&self, frame: usize) -> vk::Buffer {
        self.buffers[frame].buffer
    }

    fn destroy(&mut self, ctx: &GpuContext) {
        for b in &mut self.buffers {
            b.destroy(ctx);
        }
    }
}

/// Draws [`Overlay`]s.
#[derive(Debug)]
pub struct OverlayRenderer {
    layout: vk::PipelineLayout,
    triangles: vk::Pipeline,
    lines: vk::Pipeline,
    points: vk::Pipeline,
    triangle_buffer: StreamBuffer,
    line_buffer: StreamBuffer,
    point_buffer: StreamBuffer,
    /// Element counts written for each frame slot.
    counts: Vec<[u32; 3]>,
}

impl OverlayRenderer {
    pub fn new(
        ctx: &GpuContext,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self, RenderError> {
        let mut renderer = Self {
            layout: vk::PipelineLayout::null(),
            triangles: vk::Pipeline::null(),
            lines: vk::Pipeline::null(),
            points: vk::Pipeline::null(),
            triangle_buffer: StreamBuffer::new(
                vk::BufferUsageFlags::VERTEX_BUFFER,
                "overlay triangles",
            ),
            line_buffer: StreamBuffer::new(vk::BufferUsageFlags::VERTEX_BUFFER, "overlay lines"),
            point_buffer: StreamBuffer::new(vk::BufferUsageFlags::VERTEX_BUFFER, "overlay points"),
            counts: vec![[0; 3]; FRAMES_IN_FLIGHT],
        };
        match renderer.create_pipelines(ctx, color_format, depth_format) {
            Ok(()) => Ok(renderer),
            Err(e) => {
                renderer.destroy(ctx);
                Err(e)
            }
        }
    }

    fn create_pipelines(
        &mut self,
        ctx: &GpuContext,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<(), RenderError> {
        let device = &ctx.device;
        let code = ash::util::read_spv(&mut Cursor::new(OVERLAY_SPV))
            .map_err(|e| RenderError::Other(e.to_string()))?;
        // SAFETY: valid SPIR-V produced by slangc.
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
        };
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<OverlayPushConstants>() as u32)];
        // SAFETY: valid create info; destroyed in `destroy`.
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push),
                None,
            )
        };
        let result = layout.map_err(RenderError::from).and_then(|layout| {
            self.layout = layout;
            let f32x3 = vk::Format::R32G32B32_SFLOAT;
            let rgba8 = vk::Format::R8G8B8A8_UNORM;
            let attr = |location, format, offset: usize| {
                vk::VertexInputAttributeDescription::default()
                    .location(location)
                    .binding(0)
                    .format(format)
                    .offset(offset as u32)
            };
            let make = |vertex: &std::ffi::CStr,
                        fragment: &std::ffi::CStr,
                        stride: usize,
                        rate: vk::VertexInputRate,
                        attributes: &[vk::VertexInputAttributeDescription]| {
                PipelineDesc {
                    module,
                    vertex,
                    fragment,
                    binding: vk::VertexInputBindingDescription::default()
                        .binding(0)
                        .stride(stride as u32)
                        .input_rate(rate),
                    attributes: attributes.to_vec(),
                }
                .create(device, self.layout, color_format, depth_format)
            };
            self.triangles = make(
                c"triangleVertex",
                c"triangleFragment",
                size_of::<OverlayVertex>(),
                vk::VertexInputRate::VERTEX,
                &[
                    attr(0, f32x3, std::mem::offset_of!(OverlayVertex, position)),
                    attr(1, rgba8, std::mem::offset_of!(OverlayVertex, color)),
                    attr(
                        2,
                        vk::Format::R32G32_SFLOAT,
                        std::mem::offset_of!(OverlayVertex, uv),
                    ),
                    attr(
                        3,
                        vk::Format::R32_SFLOAT,
                        std::mem::offset_of!(OverlayVertex, checker),
                    ),
                ],
            )?;
            self.lines = make(
                c"lineVertex",
                c"colorFragment",
                size_of::<OverlayLine>(),
                vk::VertexInputRate::INSTANCE,
                &[
                    attr(0, f32x3, std::mem::offset_of!(OverlayLine, a)),
                    attr(1, f32x3, std::mem::offset_of!(OverlayLine, b)),
                    attr(2, rgba8, std::mem::offset_of!(OverlayLine, color)),
                ],
            )?;
            self.points = make(
                c"pointVertex",
                c"pointFragment",
                size_of::<OverlayPoint>(),
                vk::VertexInputRate::INSTANCE,
                &[
                    attr(0, f32x3, std::mem::offset_of!(OverlayPoint, position)),
                    attr(1, rgba8, std::mem::offset_of!(OverlayPoint, color)),
                    attr(
                        2,
                        vk::Format::R32_SFLOAT,
                        std::mem::offset_of!(OverlayPoint, size),
                    ),
                ],
            )?;
            Ok(())
        });
        // SAFETY: modules may be destroyed once pipelines are created.
        unsafe { device.destroy_shader_module(module, None) };
        result
    }

    /// Uploads the overlays for `frame`. Call after the frame's fence was
    /// waited on and before recording.
    pub fn upload(
        &mut self,
        ctx: &GpuContext,
        frame: usize,
        overlays: &[&Overlay],
    ) -> Result<(), RenderError> {
        let triangles: Vec<&[u8]> = overlays
            .iter()
            .map(|o| bytemuck::cast_slice(&o.triangles))
            .collect();
        let lines: Vec<&[u8]> = overlays
            .iter()
            .map(|o| bytemuck::cast_slice(&o.lines))
            .collect();
        let points: Vec<&[u8]> = overlays
            .iter()
            .map(|o| bytemuck::cast_slice(&o.points))
            .collect();
        self.triangle_buffer.write(ctx, frame, &triangles)?;
        self.line_buffer.write(ctx, frame, &lines)?;
        self.point_buffer.write(ctx, frame, &points)?;
        let count = |f: fn(&Overlay) -> usize| overlays.iter().map(|o| f(o)).sum::<usize>() as u32;
        self.counts[frame] = [
            count(|o| o.triangles.len()),
            count(|o| o.lines.len()),
            count(|o| o.points.len()),
        ];
        Ok(())
    }

    /// Records the overlay draws. Must be called inside the scene's rendering
    /// scope, with viewport and scissor already set to `viewport_size`.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame: usize,
        view_proj: Mat4,
        eye: Vec3,
        viewport_size: Vec2,
        style: &OverlayStyle,
    ) {
        let [triangles, lines, points] = self.counts[frame];
        if triangles + lines + points == 0 {
            return;
        }
        let push = OverlayPushConstants {
            view_proj,
            eye: eye.extend(1.0),
            viewport_size,
            line_width: style.line_width,
            depth_pull: style.surface_offset.clamp(0.0, 0.5),
        };
        let compare = if style.x_ray {
            vk::CompareOp::ALWAYS
        } else {
            vk::CompareOp::GREATER_OR_EQUAL
        };
        // SAFETY: `cmd` is recording in a compatible rendering scope; the buffers
        // of this frame slot stay untouched until its fence signals.
        unsafe {
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            if triangles > 0 {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.triangles);
                device.cmd_set_depth_compare_op(cmd, compare);
                device.cmd_bind_vertex_buffers(cmd, 0, &[self.triangle_buffer.buffer(frame)], &[0]);
                device.cmd_draw(cmd, triangles, 1, 0, 0);
            }
            if lines > 0 {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.lines);
                device.cmd_set_depth_compare_op(cmd, compare);
                device.cmd_bind_vertex_buffers(cmd, 0, &[self.line_buffer.buffer(frame)], &[0]);
                device.cmd_draw(cmd, 6, lines, 0, 0);
            }
            if points > 0 {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.points);
                device.cmd_set_depth_compare_op(cmd, compare);
                device.cmd_bind_vertex_buffers(cmd, 0, &[self.point_buffer.buffer(frame)], &[0]);
                device.cmd_draw(cmd, 6, points, 0, 0);
            }
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        // SAFETY: the caller guarantees the GPU is idle; null handles are ignored.
        unsafe {
            for p in [self.triangles, self.lines, self.points] {
                ctx.device.destroy_pipeline(p, None);
            }
            ctx.device.destroy_pipeline_layout(self.layout, None);
        }
        self.triangle_buffer.destroy(ctx);
        self.line_buffer.destroy(ctx);
        self.point_buffer.destroy(ctx);
    }
}

struct PipelineDesc<'a> {
    module: vk::ShaderModule,
    vertex: &'a std::ffi::CStr,
    fragment: &'a std::ffi::CStr,
    binding: vk::VertexInputBindingDescription,
    attributes: Vec<vk::VertexInputAttributeDescription>,
}

impl PipelineDesc<'_> {
    fn create(
        &self,
        device: &ash::Device,
        layout: vk::PipelineLayout,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<vk::Pipeline, RenderError> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.module)
                .name(self.vertex),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.module)
                .name(self.fragment),
        ];
        let bindings = [self.binding];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&self.attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Tested against the scene, but never written: overlays do not occlude each other.
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::DEPTH_COMPARE_OP,
        ];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let color_formats = [color_format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(depth_format);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        // SAFETY: all state referenced by `info` is alive for the call.
        let pipelines =
            unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) }
                .map_err(|(_, e)| e)?;
        Ok(pipelines[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_shaders() {
        assert_eq!(size_of::<OverlayVertex>(), 28);
        assert_eq!(size_of::<OverlayLine>(), 28);
        assert_eq!(size_of::<OverlayPoint>(), 20);
        assert_eq!(size_of::<OverlayPushConstants>(), 96);
    }
}
