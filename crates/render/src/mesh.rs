//! Shaded triangle mesh rendering.

use std::io::Cursor;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use polyflow_core::glam::{Mat4, Vec3, Vec4};
use polyflow_core::primitives::TriMesh;

use crate::RenderError;
use crate::context::GpuContext;
use crate::memory::GpuBuffer;

const MESH_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh.spv"));

/// Must match `PushConstants` in `shaders/mesh.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MeshPushConstants {
    pub view_proj: Mat4,
    pub eye: Vec4,
    pub color: Vec4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: Vec3,
    normal: Vec3,
}

/// A mesh uploaded to device-local memory.
#[derive(Debug)]
pub struct GpuMesh {
    vertices: GpuBuffer,
    indices: GpuBuffer,
    pub index_count: u32,
}

impl GpuMesh {
    pub fn upload(
        ctx: &GpuContext,
        command_pool: vk::CommandPool,
        mesh: &TriMesh,
    ) -> Result<Self, RenderError> {
        let vertices: Vec<Vertex> = mesh
            .positions
            .iter()
            .zip(&mesh.normals)
            .map(|(&position, &normal)| Vertex { position, normal })
            .collect();
        let index_count = u32::try_from(mesh.indices.len())
            .map_err(|_| RenderError::Other("mesh has too many indices".into()))?;
        let mut vertex_buffer = GpuBuffer::new_device_local(
            ctx,
            command_pool,
            bytemuck::cast_slice(&vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "mesh vertices",
        )?;
        let index_buffer = GpuBuffer::new_device_local(
            ctx,
            command_pool,
            bytemuck::cast_slice(&mesh.indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
            "mesh indices",
        );
        match index_buffer {
            Ok(indices) => Ok(Self {
                vertices: vertex_buffer,
                indices,
                index_count,
            }),
            Err(e) => {
                vertex_buffer.destroy(ctx);
                Err(e)
            }
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        self.vertices.destroy(ctx);
        self.indices.destroy(ctx);
    }
}

#[derive(Debug)]
pub struct MeshPipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl MeshPipeline {
    pub fn new(
        ctx: &GpuContext,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self, RenderError> {
        let device = &ctx.device;
        let code = ash::util::read_spv(&mut Cursor::new(MESH_SPV))
            .map_err(|e| RenderError::Other(e.to_string()))?;
        // SAFETY: `code` is valid SPIR-V produced by slangc.
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
        };

        let push_constants = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<MeshPushConstants>() as u32)];
        let layout_info =
            vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constants);
        // SAFETY: valid create info.
        let layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
            Ok(l) => l,
            Err(e) => {
                // SAFETY: the module is unused.
                unsafe { device.destroy_shader_module(module, None) };
                return Err(e.into());
            }
        };

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(module)
                .name(c"vertexMain"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(module)
                .name(c"fragmentMain"),
        ];
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(Vertex, normal) as u32),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        // No culling: sculpts are often open, and back faces are tinted in the shader.
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Reverse-Z: larger depth is closer.
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
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
        let result =
            unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
        // SAFETY: modules may be destroyed once pipelines are created.
        unsafe { device.destroy_shader_module(module, None) };
        match result {
            Ok(pipelines) => Ok(Self {
                layout,
                pipeline: pipelines[0],
            }),
            Err((_, e)) => {
                // SAFETY: the layout is unused.
                unsafe { device.destroy_pipeline_layout(layout, None) };
                Err(e.into())
            }
        }
    }

    /// Records the draw of `mesh`. Must be called inside dynamic rendering with
    /// attachments matching the formats given to [`MeshPipeline::new`].
    pub fn draw(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        mesh: &GpuMesh,
        push: &MeshPushConstants,
    ) {
        // SAFETY: `cmd` is recording inside a compatible rendering scope and the
        // mesh buffers stay alive until the frame's fence is signaled.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(push),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertices.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, mesh.indices.buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        // SAFETY: the caller guarantees the GPU is idle.
        unsafe {
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
