//! Reference mesh rendering with GPU meshlet culling.
//!
//! The mesh is split into meshlets of up to [`MESHLET_MAX_TRIANGLES`]
//! triangles. Each frame a compute pass tests every meshlet against the view
//! frustum and (optionally) its normal cone for backfacing, and appends the
//! survivors as indexed indirect draws; the graphics pass then draws them all
//! with a single `vkCmdDrawIndexedIndirectCount`.

use std::io::Cursor;
use std::time::{Duration, Instant};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use polyflow_core::glam::{Mat4, Vec3, Vec4};
use polyflow_core::mesh::TriMesh;
use rayon::prelude::*;

use crate::RenderError;
use crate::context::GpuContext;
use crate::memory::GpuBuffer;
use crate::renderer::FRAMES_IN_FLIGHT;

const MESH_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh.spv"));
const CULL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cull.spv"));

/// NVIDIA's recommended meshlet size, also a good fit for AMD.
pub const MESHLET_MAX_VERTICES: usize = 64;
pub const MESHLET_MAX_TRIANGLES: usize = 124;
/// Favors tighter normal cones (better backface culling) over spatial compactness.
const MESHLET_CONE_WEIGHT: f32 = 0.25;
/// Triangles per independently clusterized block, so meshlets are built on
/// all cores. Blocks follow the index order, which is spatially coherent in
/// sculpt exports.
const BUILD_BLOCK_TRIANGLES: usize = 1 << 18;
const CULL_WORKGROUP_SIZE: u32 = 64;

const CULL_FLAG_BACKFACE: u32 = 1;
const MESH_FLAG_SHOW_MESHLETS: u32 = 1;

/// A vertex as stored on the GPU: 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
struct PackedVertex {
    position: [f32; 3],
    /// Octahedral-encoded unit normal, `R16G16_SNORM`.
    normal: [i16; 2],
}

/// Must match `Meshlet` in `shaders/cull.slang` (std430, 48 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
struct GpuMeshlet {
    center: [f32; 3],
    radius: f32,
    cone_axis: [f32; 3],
    cone_cutoff: f32,
    first_index: u32,
    index_count: u32,
    pad: [u32; 2],
}

/// `VkDrawIndexedIndirectCommand`.
const DRAW_COMMAND_SIZE: u64 = 20;

/// Must match `CullConstants` in `shaders/cull.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CullPushConstants {
    planes: [Vec4; 5],
    eye: Vec4,
    meshlets: u64,
    draws: u64,
    counters: u64,
    meshlet_count: u32,
    flags: u32,
}

/// Must match `PushConstants` in `shaders/mesh.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MeshPushConstants {
    view_proj: Mat4,
    eye: Vec4,
    color: Vec4,
    flags: u32,
    pad: [u32; 3],
}

/// A mesh converted to the GPU layout (packed vertices, meshlets), ready to
/// upload. Building it is CPU-heavy, so do it off the main thread.
#[derive(Debug, Default)]
pub struct MeshData {
    vertices: Vec<PackedVertex>,
    /// Global vertex indices, grouped by meshlet.
    indices: Vec<u32>,
    meshlets: Vec<GpuMeshlet>,
    /// Time taken by [`MeshData::build`].
    pub build_time: Duration,
}

impl MeshData {
    /// Packs vertices and splits the mesh into meshlets, in parallel.
    pub fn build(mesh: &TriMesh) -> Self {
        let start = Instant::now();
        let vertices: Vec<PackedVertex> = mesh
            .positions
            .par_iter()
            .zip(&mesh.normals)
            .map(|(p, &n)| PackedVertex {
                position: p.to_array(),
                normal: encode_octahedral(n),
            })
            .collect();

        let positions: &[u8] = bytemuck::cast_slice(&mesh.positions);
        let blocks: Vec<(Vec<u32>, Vec<GpuMeshlet>)> = mesh
            .indices
            .par_chunks(BUILD_BLOCK_TRIANGLES * 3)
            .map(|block| build_block(block, positions))
            .collect();

        // Concatenate blocks in parallel, rebasing each block's index offsets.
        let mut index_offsets = Vec::with_capacity(blocks.len());
        let (mut index_count, mut meshlet_count) = (0, 0);
        for (indices, meshlets) in &blocks {
            index_offsets.push(index_count);
            index_count += indices.len();
            meshlet_count += meshlets.len();
        }
        let mut indices = vec![0u32; index_count];
        let mut meshlets = vec![GpuMeshlet::zeroed(); meshlet_count];
        let index_slices = split_mut(&mut indices, blocks.iter().map(|b| b.0.len()));
        let meshlet_slices = split_mut(&mut meshlets, blocks.iter().map(|b| b.1.len()));
        blocks
            .par_iter()
            .zip(index_slices)
            .zip(meshlet_slices)
            .zip(&index_offsets)
            .for_each(
                |((((block_indices, block_meshlets), indices), meshlets), &offset)| {
                    indices.copy_from_slice(block_indices);
                    for (dst, src) in meshlets.iter_mut().zip(block_meshlets) {
                        *dst = GpuMeshlet {
                            first_index: src.first_index + offset as u32,
                            ..*src
                        };
                    }
                },
            );

        Self {
            vertices,
            indices,
            meshlets,
            build_time: start.elapsed(),
        }
    }

    pub fn meshlet_count(&self) -> usize {
        self.meshlets.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Clusterizes one block of triangles. Returned `first_index` values are
/// relative to the block.
fn build_block(indices: &[u32], positions: &[u8]) -> (Vec<u32>, Vec<GpuMeshlet>) {
    let adapter = meshopt::VertexDataAdapter::new(positions, size_of::<Vec3>(), 0)
        .expect("positions are tightly packed Vec3s");
    let built = meshopt::build_meshlets(
        indices,
        &adapter,
        MESHLET_MAX_VERTICES,
        MESHLET_MAX_TRIANGLES,
        MESHLET_CONE_WEIGHT,
    );
    let mut out_indices = Vec::with_capacity(indices.len());
    let mut out_meshlets = Vec::with_capacity(built.len());
    for meshlet in built.iter() {
        let bounds = meshopt::compute_meshlet_bounds(meshlet, &adapter);
        let first_index = out_indices.len() as u32;
        out_indices.extend(
            meshlet
                .triangles
                .iter()
                .map(|&local| meshlet.vertices[local as usize]),
        );
        out_meshlets.push(GpuMeshlet {
            center: bounds.center,
            radius: bounds.radius,
            cone_axis: bounds.cone_axis,
            cone_cutoff: bounds.cone_cutoff,
            first_index,
            index_count: meshlet.triangles.len() as u32,
            pad: [0; 2],
        });
    }
    (out_indices, out_meshlets)
}

/// Splits `slice` into consecutive mutable pieces of the given lengths.
fn split_mut<T>(mut slice: &mut [T], lengths: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    lengths
        .map(|len| {
            let (head, tail) = std::mem::take(&mut slice).split_at_mut(len);
            slice = tail;
            head
        })
        .collect()
}

/// Octahedral encoding of a unit vector into two snorm16 values.
fn encode_octahedral(n: Vec3) -> [i16; 2] {
    let sign = |v: f32| if v >= 0.0 { 1.0 } else { -1.0 };
    let n = n / (n.x.abs() + n.y.abs() + n.z.abs()).max(f32::MIN_POSITIVE);
    let (x, y) = if n.z >= 0.0 {
        (n.x, n.y)
    } else {
        ((1.0 - n.y.abs()) * sign(n.x), (1.0 - n.x.abs()) * sign(n.y))
    };
    [x, y].map(|v| (v.clamp(-1.0, 1.0) * 32767.0).round() as i16)
}

/// Visibility statistics of one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CullStats {
    pub visible_meshlets: u32,
    pub visible_triangles: u32,
    pub total_meshlets: u32,
}

/// Per-frame-in-flight buffers written by the culling pass.
#[derive(Debug)]
struct CullBuffers {
    draws: GpuBuffer,
    /// `[draw count, visible triangles]`.
    counters: GpuBuffer,
    /// Host-visible copy of `counters`, read once the frame has completed.
    readback: GpuBuffer,
    has_stats: bool,
}

/// A mesh uploaded to the GPU, with its culling buffers.
#[derive(Debug)]
pub struct GpuMesh {
    vertices: GpuBuffer,
    indices: GpuBuffer,
    meshlets: GpuBuffer,
    meshlet_count: u32,
    frames: Vec<CullBuffers>,
}

impl GpuMesh {
    pub fn upload(
        ctx: &GpuContext,
        command_pool: vk::CommandPool,
        data: &MeshData,
    ) -> Result<Self, RenderError> {
        let meshlet_count = u32::try_from(data.meshlets.len())
            .map_err(|_| RenderError::Other("too many meshlets".into()))?;
        let max_draws = ctx.properties.limits.max_draw_indirect_count;
        if meshlet_count > max_draws {
            return Err(RenderError::Other(format!(
                "the mesh has {meshlet_count} meshlets but this GPU can draw at most {max_draws} per call"
            )));
        }
        let mut mesh = Self {
            vertices: GpuBuffer::null(),
            indices: GpuBuffer::null(),
            meshlets: GpuBuffer::null(),
            meshlet_count,
            frames: vec![],
        };
        match mesh.create_buffers(ctx, command_pool, data) {
            Ok(()) => Ok(mesh),
            Err(e) => {
                mesh.destroy(ctx);
                Err(e)
            }
        }
    }

    fn create_buffers(
        &mut self,
        ctx: &GpuContext,
        command_pool: vk::CommandPool,
        data: &MeshData,
    ) -> Result<(), RenderError> {
        use vk::BufferUsageFlags as U;
        self.vertices = GpuBuffer::new_device_local(
            ctx,
            command_pool,
            bytemuck::cast_slice(&data.vertices),
            U::VERTEX_BUFFER,
            "mesh vertices",
        )?;
        self.indices = GpuBuffer::new_device_local(
            ctx,
            command_pool,
            bytemuck::cast_slice(&data.indices),
            U::INDEX_BUFFER,
            "mesh indices",
        )?;
        self.meshlets = GpuBuffer::new_device_local(
            ctx,
            command_pool,
            bytemuck::cast_slice(&data.meshlets),
            U::STORAGE_BUFFER | U::SHADER_DEVICE_ADDRESS,
            "meshlets",
        )?;
        for _ in 0..FRAMES_IN_FLIGHT {
            let mut frame = CullBuffers {
                draws: GpuBuffer::null(),
                counters: GpuBuffer::null(),
                readback: GpuBuffer::null(),
                has_stats: false,
            };
            let created = (|| {
                frame.draws = GpuBuffer::new(
                    ctx,
                    u64::from(self.meshlet_count) * DRAW_COMMAND_SIZE,
                    U::STORAGE_BUFFER | U::INDIRECT_BUFFER | U::SHADER_DEVICE_ADDRESS,
                    MemoryLocation::GpuOnly,
                    "meshlet draws",
                )?;
                frame.counters = GpuBuffer::new(
                    ctx,
                    8,
                    U::STORAGE_BUFFER
                        | U::INDIRECT_BUFFER
                        | U::SHADER_DEVICE_ADDRESS
                        | U::TRANSFER_DST
                        | U::TRANSFER_SRC,
                    MemoryLocation::GpuOnly,
                    "cull counters",
                )?;
                frame.readback = GpuBuffer::new(
                    ctx,
                    8,
                    U::TRANSFER_DST,
                    MemoryLocation::GpuToCpu,
                    "cull readback",
                )?;
                Ok::<_, RenderError>(())
            })();
            // Pushed even on failure so `destroy` releases what was created.
            self.frames.push(frame);
            created?;
        }
        Ok(())
    }

    /// Statistics recorded the last time `frame` was rendered. Call after the
    /// frame's fence has been waited on.
    pub fn take_stats(&mut self, frame: usize) -> Option<CullStats> {
        let buffers = self.frames.get_mut(frame)?;
        if !std::mem::take(&mut buffers.has_stats) {
            return None;
        }
        let counters: &[u32] =
            bytemuck::try_cast_slice(buffers.readback.mapped()?.get(..8)?).ok()?;
        Some(CullStats {
            visible_meshlets: counters[0],
            visible_triangles: counters[1],
            total_meshlets: self.meshlet_count,
        })
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        self.vertices.destroy(ctx);
        self.indices.destroy(ctx);
        self.meshlets.destroy(ctx);
        for frame in &mut self.frames {
            frame.draws.destroy(ctx);
            frame.counters.destroy(ctx);
            frame.readback.destroy(ctx);
        }
    }
}

/// Camera and display settings for drawing a mesh.
#[derive(Debug, Clone, Copy)]
pub struct MeshView {
    pub view_proj: Mat4,
    pub eye: Vec3,
    pub color: Vec3,
    pub backface_culling: bool,
    pub show_meshlets: bool,
}

/// The culling compute pipeline and the mesh graphics pipeline.
#[derive(Debug)]
pub struct MeshPipeline {
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cull_layout: vk::PipelineLayout,
    cull_pipeline: vk::Pipeline,
}

impl MeshPipeline {
    pub fn new(
        ctx: &GpuContext,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self, RenderError> {
        let mut pipeline = Self {
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            cull_layout: vk::PipelineLayout::null(),
            cull_pipeline: vk::Pipeline::null(),
        };
        match pipeline.create(ctx, color_format, depth_format) {
            Ok(()) => Ok(pipeline),
            Err(e) => {
                pipeline.destroy(ctx);
                Err(e)
            }
        }
    }

    fn create(
        &mut self,
        ctx: &GpuContext,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<(), RenderError> {
        let device = &ctx.device;

        let cull_module = create_shader_module(device, CULL_SPV)?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size_of::<CullPushConstants>() as u32)];
        // SAFETY: valid create info; the layout is destroyed in `destroy`.
        self.cull_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push),
                None,
            )
        }
        .inspect_err(|_| unsafe { device.destroy_shader_module(cull_module, None) })?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(cull_module)
            .name(c"cullMain");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.cull_layout);
        // SAFETY: valid create info; the module is no longer needed afterwards.
        let result =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        // SAFETY: modules may be destroyed once pipelines are created.
        unsafe { device.destroy_shader_module(cull_module, None) };
        self.cull_pipeline = result.map_err(|(_, e)| e)?[0];

        let module = create_shader_module(device, MESH_SPV)?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<MeshPushConstants>() as u32)];
        // SAFETY: valid create info; the layout is destroyed in `destroy`.
        self.layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push),
                None,
            )
        }
        .inspect_err(|_| unsafe { device.destroy_shader_module(module, None) })?;

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
            .stride(size_of::<PackedVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attributes = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(PackedVertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .format(vk::Format::R16G16_SNORM)
                .offset(std::mem::offset_of!(PackedVertex, normal) as u32),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        // Cull mode is dynamic: back faces are culled or tinted depending on the view settings.
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
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
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::CULL_MODE,
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
            .layout(self.layout)
            .push_next(&mut rendering);
        // SAFETY: all state referenced by `info` is alive for the call.
        let result =
            unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
        // SAFETY: modules may be destroyed once pipelines are created.
        unsafe { device.destroy_shader_module(module, None) };
        self.pipeline = result.map_err(|(_, e)| e)?[0];
        Ok(())
    }

    /// Records the culling pass for `frame`. Must be recorded outside a
    /// rendering scope, before [`MeshPipeline::draw`] for the same frame.
    pub fn record_cull(
        &self,
        ctx: &GpuContext,
        cmd: vk::CommandBuffer,
        mesh: &mut GpuMesh,
        frame: usize,
        view: &MeshView,
    ) {
        let device = &ctx.device;
        let meshlets_address = mesh.meshlets.device_address(ctx);
        let buffers = &mut mesh.frames[frame];
        let constants = CullPushConstants {
            planes: polyflow_core::camera::frustum_planes(view.view_proj),
            eye: view.eye.extend(1.0),
            meshlets: meshlets_address,
            draws: buffers.draws.device_address(ctx),
            counters: buffers.counters.device_address(ctx),
            meshlet_count: mesh.meshlet_count,
            flags: if view.backface_culling {
                CULL_FLAG_BACKFACE
            } else {
                0
            },
        };
        // SAFETY: `cmd` is recording outside a rendering scope; the buffers belong
        // to this frame slot, whose previous use completed (its fence was waited on).
        unsafe {
            device.cmd_fill_buffer(cmd, buffers.counters.buffer, 0, 8, 0);
            let cleared = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                )
                .buffer(buffers.counters.buffer)
                .size(vk::WHOLE_SIZE)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&cleared),
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline);
            device.cmd_push_constants(
                cmd,
                self.cull_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&constants),
            );
            device.cmd_dispatch(cmd, mesh.meshlet_count.div_ceil(CULL_WORKGROUP_SIZE), 1, 1);

            let culled = [
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT)
                    .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                    .buffer(buffers.draws.buffer)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::COPY,
                    )
                    .dst_access_mask(
                        vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::TRANSFER_READ,
                    )
                    .buffer(buffers.counters.buffer)
                    .size(vk::WHOLE_SIZE),
            ];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&culled),
            );
            let region = vk::BufferCopy::default().size(8);
            device.cmd_copy_buffer(
                cmd,
                buffers.counters.buffer,
                buffers.readback.buffer,
                &[region],
            );
            // Makes the copy visible to the host once the frame's fence signals.
            let to_host = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)
                .buffer(buffers.readback.buffer)
                .size(vk::WHOLE_SIZE)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&to_host),
            );
        }
        buffers.has_stats = true;
    }

    /// Draws the meshlets that survived culling. Must be called inside
    /// dynamic rendering with attachments matching the formats given to
    /// [`MeshPipeline::new`], with viewport and scissor already set.
    pub fn draw(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        mesh: &GpuMesh,
        frame: usize,
        view: &MeshView,
    ) {
        let push = MeshPushConstants {
            view_proj: view.view_proj,
            eye: view.eye.extend(1.0),
            color: view.color.extend(1.0),
            flags: if view.show_meshlets {
                MESH_FLAG_SHOW_MESHLETS
            } else {
                0
            },
            pad: [0; 3],
        };
        let cull_mode = if view.backface_culling {
            vk::CullModeFlags::BACK
        } else {
            vk::CullModeFlags::NONE
        };
        let buffers = &mesh.frames[frame];
        // SAFETY: `cmd` is recording inside a compatible rendering scope; the
        // buffers stay alive until the frame's fence signals.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_set_cull_mode(cmd, cull_mode);
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertices.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, mesh.indices.buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed_indirect_count(
                cmd,
                buffers.draws.buffer,
                0,
                buffers.counters.buffer,
                0,
                mesh.meshlet_count,
                DRAW_COMMAND_SIZE as u32,
            );
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        // SAFETY: the caller guarantees the GPU is idle; null handles are ignored.
        unsafe {
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device.destroy_pipeline_layout(self.layout, None);
            ctx.device.destroy_pipeline(self.cull_pipeline, None);
            ctx.device.destroy_pipeline_layout(self.cull_layout, None);
        }
    }
}

fn create_shader_module(device: &ash::Device, spv: &[u8]) -> Result<vk::ShaderModule, RenderError> {
    let code = ash::util::read_spv(&mut Cursor::new(spv))
        .map_err(|e| RenderError::Other(e.to_string()))?;
    // SAFETY: `code` is valid SPIR-V produced by slangc.
    Ok(unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
    })
}

#[cfg(test)]
mod tests {
    use polyflow_core::primitives::torus;

    use super::*;

    fn decode_octahedral(e: [i16; 2]) -> Vec3 {
        let (x, y) = (f32::from(e[0]) / 32767.0, f32::from(e[1]) / 32767.0);
        let mut n = Vec3::new(x, y, 1.0 - x.abs() - y.abs());
        let t = (-n.z).clamp(0.0, 1.0);
        n.x += if n.x >= 0.0 { -t } else { t };
        n.y += if n.y >= 0.0 { -t } else { t };
        n.normalize()
    }

    #[test]
    fn octahedral_round_trip() {
        let mesh = torus(1.0, 0.3, 64, 32);
        for &n in mesh.normals.iter().chain(&[
            Vec3::X,
            Vec3::NEG_Y,
            Vec3::NEG_Z,
            Vec3::new(-1.0, -1.0, -1.0).normalize(),
        ]) {
            let decoded = decode_octahedral(encode_octahedral(n));
            assert!(decoded.dot(n) > 0.99999, "{n} -> {decoded}");
        }
    }

    #[test]
    fn layouts_match_shaders() {
        assert_eq!(size_of::<PackedVertex>(), 16);
        assert_eq!(size_of::<GpuMeshlet>(), 48);
        assert_eq!(size_of::<CullPushConstants>(), 128);
        assert!(size_of::<MeshPushConstants>() <= 128);
    }

    #[test]
    fn meshlets_cover_every_triangle_once() {
        let mesh = torus(1.0, 0.3, 300, 200);
        let data = MeshData::build(&mesh);
        assert_eq!(data.triangle_count(), mesh.triangle_count());
        assert_eq!(data.vertices.len(), mesh.vertex_count());

        let mut expected: Vec<[u32; 3]> = mesh
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| canonical(*t))
            .collect();
        let mut actual = Vec::with_capacity(expected.len());
        let mut next_index = 0;
        for m in &data.meshlets {
            assert_eq!(m.first_index, next_index, "meshlets are contiguous");
            assert!(m.index_count as usize <= MESHLET_MAX_TRIANGLES * 3);
            let range = m.first_index as usize..(m.first_index + m.index_count) as usize;
            for tri in data.indices[range].as_chunks::<3>().0 {
                actual.push(canonical(*tri));
                // Every vertex of the meshlet lies inside its bounding sphere.
                for &i in tri {
                    let p = mesh.positions[i as usize];
                    assert!(p.distance(Vec3::from_array(m.center)) <= m.radius * 1.0001 + 1e-6);
                }
            }
            next_index += m.index_count;
        }
        expected.sort_unstable();
        actual.sort_unstable();
        assert_eq!(actual, expected);
    }

    /// Rotates a triangle so its smallest index comes first, keeping the winding.
    fn canonical([a, b, c]: [u32; 3]) -> [u32; 3] {
        let min = a.min(b).min(c);
        if min == a {
            [a, b, c]
        } else if min == b {
            [b, c, a]
        } else {
            [c, a, b]
        }
    }
}
