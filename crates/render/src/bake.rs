//! The GPU baker: `polyflow_bake::cpu` running in a compute shader.
//!
//! Rays are traced with hardware ray queries when the GPU has them (an
//! acceleration structure built from the reference mesh), or else through
//! the reference's own BVH in plain compute. Both give the same results to
//! within float rounding.
//!
//! The reference mesh, its BVH and the low-poly are uploaded once. The map is
//! then processed in bands of rows, each small enough that no dispatch runs
//! long (operating systems reset GPUs that stay busy for about two seconds).
//! For every band and sample, the CPU rasterizes the UVs (cheap) and the GPU
//! traces the rays, adding into per-texel sums that stay on the GPU until
//! the band is done. The sums then go through the same
//! [`polyflow_bake::finish`] as the CPU baker's, so both produce maps the
//! same way.

use std::io::Cursor;

use ash::{khr, vk};
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use polyflow_bake::{
    BakeControl, BakeMesh, BakeSettings, BakeSums, Baked, MAX_SAMPLES, Reference, rasterize_rows,
};
use polyflow_core::glam::Vec2;

use crate::RenderError;
use crate::context::GpuContext;
use crate::memory::{GpuBuffer, submit_and_wait};

const BAKE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bake.spv"));
const BAKE_RT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bake_rt.spv"));
/// Threads per workgroup, as in `bake.slang`.
const WORKGROUP: u32 = 64;
/// Rays per dispatch to aim for: a fraction of a second on a mid-range GPU.
const RAYS_PER_DISPATCH: u64 = 1 << 25;

const MAP_TANGENT: u32 = 1;
const MAP_OBJECT: u32 = 2;
const MAP_AO: u32 = 4;
const FLAG_DIRECTX: u32 = 8;
const FLAG_CAGE: u32 = 16;

/// Must match `Params` in `shaders/bake.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BakeParams {
    corners: u64,
    ref_positions: u64,
    ref_normals: u64,
    ref_indices: u64,
    nodes: u64,
    node_triangles: u64,
    ao_points: u64,
    raster_triangles: u64,
    raster_bary: u64,
    sum_tangent: u64,
    sum_object: u64,
    sum_ao: u64,
    hits: u64,
    covered: u64,
    width: u32,
    band_rows: u32,
    flags: u32,
    ao_count: u32,
    frontal: f32,
    rear: f32,
    ao_distance: f32,
    ao_bias: f32,
}

/// Must match `PushConstants` in `shaders/bake.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BakePushConstants {
    params: u64,
    pass: u32,
    first_row: u32,
}

/// How the GPU traces rays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tracer {
    /// Hardware ray queries if the GPU has them, else [`Tracer::Compute`].
    #[default]
    Auto,
    /// Through the reference mesh's BVH, in plain compute: any GPU.
    Compute,
    /// Hardware ray queries against an acceleration structure.
    RayQuery,
}

impl Tracer {
    /// What `self` means on `ctx`'s GPU.
    pub fn resolve(self, ctx: &GpuContext) -> Tracer {
        match self {
            Tracer::Auto if ctx.ray_query => Tracer::RayQuery,
            Tracer::Auto => Tracer::Compute,
            other => other,
        }
    }
}

/// Why a GPU bake stopped.
#[derive(Debug, thiserror::Error)]
pub enum GpuBakeError {
    #[error(transparent)]
    Render(#[from] RenderError),
    #[error("the bake was cancelled")]
    Cancelled,
}

impl From<vk::Result> for GpuBakeError {
    fn from(e: vk::Result) -> Self {
        Self::Render(e.into())
    }
}

/// Vulkan objects of one bake, released on drop.
struct Resources<'a> {
    ctx: &'a GpuContext,
    pool: vk::CommandPool,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    buffers: Vec<GpuBuffer>,
    /// Ray query path only.
    set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    accel: Option<khr::acceleration_structure::Device>,
    structures: Vec<vk::AccelerationStructureKHR>,
}

impl Drop for Resources<'_> {
    fn drop(&mut self) {
        let device = &self.ctx.device;
        // SAFETY: every submission was waited on, so nothing is in use; null
        // handles are ignored.
        unsafe {
            let _ = device.device_wait_idle();
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_command_pool(self.pool, None);
            if let Some(accel) = &self.accel {
                for &structure in &self.structures {
                    accel.destroy_acceleration_structure(structure, None);
                }
            }
        }
        for mut buffer in self.buffers.drain(..) {
            buffer.destroy(self.ctx);
        }
    }
}

impl Resources<'_> {
    /// Keeps `buffer` for release and returns its index.
    fn keep(&mut self, buffer: GpuBuffer) -> usize {
        self.buffers.push(buffer);
        self.buffers.len() - 1
    }

    /// A device-local storage buffer holding `data`; returns its address.
    fn upload(&mut self, data: &[u8], name: &str) -> Result<u64, RenderError> {
        self.upload_with(data, vk::BufferUsageFlags::empty(), name)
    }

    /// Like [`Self::upload`], with `extra` usage.
    fn upload_with(
        &mut self,
        data: &[u8],
        extra: vk::BufferUsageFlags,
        name: &str,
    ) -> Result<u64, RenderError> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | extra;
        // Zero-sized buffers are not allowed; the shader never reads empty ones.
        let data = if data.is_empty() { &[0u8; 4][..] } else { data };
        let buffer = GpuBuffer::new_device_local(self.ctx, self.pool, data, usage, name)?;
        let address = buffer.device_address(self.ctx);
        self.keep(buffer);
        Ok(address)
    }

    /// A buffer of `bytes` at `location`; returns its index.
    fn buffer(
        &mut self,
        bytes: usize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<usize, RenderError> {
        let buffer = GpuBuffer::new(self.ctx, bytes.max(4) as u64, usage, location, name)?;
        Ok(self.keep(buffer))
    }

    /// Frees buffer `index` early (it must no longer be in use).
    fn free(&mut self, index: usize) {
        let mut buffer = std::mem::replace(&mut self.buffers[index], GpuBuffer::null());
        buffer.destroy(self.ctx);
    }
}

/// Per-band GPU sums and the host-visible buffers they are read back into.
struct BandBuffers {
    /// Indices into `Resources::buffers`: (GPU sum, readback), per output.
    tangent: (usize, usize),
    object: (usize, usize),
    ao: (usize, usize),
    hits: (usize, usize),
    covered: (usize, usize),
    raster_triangles: usize,
    raster_bary: usize,
    params: usize,
}

/// Bakes `settings.maps` from `reference` onto `low` on the GPU. Gives the
/// same maps as [`polyflow_bake::bake`], to within float rounding.
pub fn bake(
    ctx: &GpuContext,
    low: &BakeMesh,
    reference: Reference<'_>,
    settings: &BakeSettings,
    control: &BakeControl,
) -> Result<Baked, GpuBakeError> {
    bake_with(ctx, low, reference, settings, control, Tracer::Auto)
}

/// [`bake`] with a choice of how rays are traced.
pub fn bake_with(
    ctx: &GpuContext,
    low: &BakeMesh,
    reference: Reference<'_>,
    settings: &BakeSettings,
    control: &BakeControl,
    tracer: Tracer,
) -> Result<Baked, GpuBakeError> {
    let ray_query = match tracer.resolve(ctx) {
        Tracer::RayQuery if !ctx.ray_query => {
            return Err(RenderError::Other("this GPU has no hardware ray tracing".into()).into());
        }
        t => t == Tracer::RayQuery,
    };
    let mut sums = BakeSums::new(settings);
    if low.triangle_count() == 0 || reference.mesh.triangle_count() == 0 {
        return Ok(polyflow_bake::finish(sums, settings));
    }
    let device = &ctx.device;
    let mut res = Resources {
        ctx,
        pool: vk::CommandPool::null(),
        layout: vk::PipelineLayout::null(),
        pipeline: vk::Pipeline::null(),
        buffers: Vec::new(),
        set_layout: vk::DescriptorSetLayout::null(),
        descriptor_pool: vk::DescriptorPool::null(),
        accel: ray_query.then(|| khr::acceleration_structure::Device::new(&ctx.instance, device)),
        structures: Vec::new(),
    };
    let pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(ctx.queue_family)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    // SAFETY: valid create info; destroyed with `res`.
    res.pool = unsafe { device.create_command_pool(&pool_info, None)? };
    let descriptor_set = if ray_query {
        Some(create_descriptor_set(&mut res)?)
    } else {
        None
    };
    create_pipeline(&mut res, ray_query)?;

    // Inputs that do not change during the bake.
    let corners: Vec<[f32; 20]> = low
        .corners
        .iter()
        .map(|c| {
            let [px, py, pz] = c.position.to_array();
            let [nx, ny, nz] = c.normal.to_array();
            let [cx, cy, cz] = c.cage.to_array();
            let [rx, ry, rz] = c.projection.to_array();
            let t = c.tangent.to_array();
            [
                px, py, pz, 0.0, nx, ny, nz, 0.0, t[0], t[1], t[2], t[3], cx, cy, cz, 0.0, rx, ry,
                rz, 0.0,
            ]
        })
        .collect();
    let mesh = reference.mesh;
    let positions: Vec<[f32; 4]> = mesh
        .positions
        .iter()
        .map(|p| p.extend(0.0).to_array())
        .collect();
    let normals: Vec<[f32; 4]> = mesh
        .normals
        .iter()
        .map(|n| n.extend(0.0).to_array())
        .collect();
    let ao_points: Vec<[f32; 2]> = polyflow_bake::cpu::ao_points(settings)
        .iter()
        .map(|p| p.to_array())
        .collect();
    // Acceleration structures read positions and indices too.
    let build_input = if ray_query {
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
    } else {
        vk::BufferUsageFlags::empty()
    };
    let mut params = BakeParams {
        corners: res.upload(bytemuck::cast_slice(&corners), "bake corners")?,
        ref_positions: res.upload_with(
            bytemuck::cast_slice(&positions),
            build_input,
            "bake positions",
        )?,
        ref_normals: res.upload(bytemuck::cast_slice(&normals), "bake normals")?,
        ref_indices: res.upload_with(
            bytemuck::cast_slice(&mesh.indices),
            build_input,
            "bake indices",
        )?,
        ao_points: res.upload(bytemuck::cast_slice(&ao_points), "bake ao points")?,
        width: settings.width,
        frontal: settings.frontal_distance,
        rear: settings.rear_distance,
        ao_distance: settings.ao_distance,
        ao_bias: polyflow_bake::cpu::ao_bias(mesh),
        ..BakeParams::zeroed()
    };
    drop((corners, positions, normals));
    match descriptor_set {
        Some(set) => {
            let scene = build_scene(&mut res, &params, mesh)?;
            write_scene(&res, set, scene);
        }
        None => {
            params.nodes =
                res.upload(bytemuck::cast_slice(&reference.bvh.gpu_nodes()), "bake bvh")?;
            params.node_triangles = res.upload(
                bytemuck::cast_slice(reference.bvh.triangle_order()),
                "bake bvh triangles",
            )?;
        }
    }
    let maps = settings.maps;
    params.flags = [
        (maps.tangent_normal, MAP_TANGENT),
        (maps.object_normal, MAP_OBJECT),
        (maps.ambient_occlusion, MAP_AO),
        (
            settings.normal_y == polyflow_bake::NormalY::DirectX,
            FLAG_DIRECTX,
        ),
        (low.has_cage, FLAG_CAGE),
    ]
    .into_iter()
    .filter(|&(on, _)| on)
    .fold(0, |flags, (_, bit)| flags | bit);
    params.ao_count = if maps.ambient_occlusion {
        ao_points.len() as u32
    } else {
        0
    };

    // Bands sized so a dispatch traces about RAYS_PER_DISPATCH rays.
    let width = settings.width as usize;
    let rays_per_texel = 1 + u64::from(params.ao_count);
    let band_texels = (RAYS_PER_DISPATCH / rays_per_texel).clamp(1 << 14, 1 << 22) as usize;
    let band_rows = (band_texels / width).clamp(1, settings.height as usize);
    let texels = band_rows * width;
    let band = band_buffers(&mut res, texels, maps)?;
    for (field, (gpu, _)) in [
        (&mut params.sum_tangent, band.tangent),
        (&mut params.sum_object, band.object),
        (&mut params.sum_ao, band.ao),
        (&mut params.hits, band.hits),
        (&mut params.covered, band.covered),
    ] {
        *field = res.buffers[gpu].device_address(ctx);
    }
    params.raster_triangles = res.buffers[band.raster_triangles].device_address(ctx);
    params.raster_bary = res.buffers[band.raster_bary].device_address(ctx);
    let params_address = res.buffers[band.params].device_address(ctx);

    let n = settings.samples.clamp(1, MAX_SAMPLES);
    let mut first_row = 0;
    while first_row < settings.height {
        let rows = first_row..(first_row + band_rows as u32).min(settings.height);
        let band_texels = rows.len() * width;
        params.band_rows = rows.len() as u32;
        write(&mut res, band.params, bytemuck::bytes_of(&params))?;
        for pass in 0..n * n {
            if control.cancel_requested() {
                return Err(GpuBakeError::Cancelled);
            }
            let offset = Vec2::new(
                ((pass % n) as f32 + 0.5) / n as f32,
                ((pass / n) as f32 + 0.5) / n as f32,
            );
            let raster = rasterize_rows(low, settings.width, settings.height, offset, rows.clone());
            write(
                &mut res,
                band.raster_triangles,
                bytemuck::cast_slice(&raster.triangles),
            )?;
            write(
                &mut res,
                band.raster_bary,
                bytemuck::cast_slice(&raster.barycentrics),
            )?;
            let push = BakePushConstants {
                params: params_address,
                pass,
                first_row: rows.start,
            };
            let clear: Vec<vk::Buffer> = if pass == 0 {
                [band.tangent, band.object, band.ao, band.hits, band.covered]
                    .iter()
                    .map(|&(gpu, _)| res.buffers[gpu].buffer)
                    .collect()
            } else {
                Vec::new()
            };
            submit_and_wait(ctx, res.pool, |cmd| {
                // SAFETY: `cmd` is recording; every buffer is alive and sized
                // for the band; the pipeline and layout match.
                unsafe {
                    for &buffer in &clear {
                        device.cmd_fill_buffer(cmd, buffer, 0, vk::WHOLE_SIZE, 0);
                    }
                    barrier(
                        device,
                        cmd,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        vk::PipelineStageFlags2::COMPUTE_SHADER,
                    );
                    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, res.pipeline);
                    if let Some(set) = descriptor_set {
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            res.layout,
                            0,
                            &[set],
                            &[],
                        );
                    }
                    device.cmd_push_constants(
                        cmd,
                        res.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&push),
                    );
                    device.cmd_dispatch(cmd, (band_texels as u32).div_ceil(WORKGROUP), 1, 1);
                }
            })?;
            control.add_rows(rows.len() as u32);
        }

        // Read the band's sums back.
        submit_and_wait(ctx, res.pool, |cmd| {
            // SAFETY: as above; sum and readback buffers have equal sizes.
            unsafe {
                barrier(
                    device,
                    cmd,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::PipelineStageFlags2::TRANSFER,
                );
                for (gpu, readback) in [band.tangent, band.object, band.ao, band.hits, band.covered]
                {
                    let size = res.buffers[gpu].size();
                    device.cmd_copy_buffer(
                        cmd,
                        res.buffers[gpu].buffer,
                        res.buffers[readback].buffer,
                        &[vk::BufferCopy::default().size(size)],
                    );
                }
                barrier(
                    device,
                    cmd,
                    vk::PipelineStageFlags2::TRANSFER,
                    vk::PipelineStageFlags2::HOST,
                );
            }
        })?;
        let start = rows.start as usize * width;
        let read = |index: usize| -> Result<&[u8], RenderError> {
            res.buffers[index]
                .mapped()
                .ok_or_else(|| RenderError::Other("readback buffer is not host visible".into()))
        };
        for (map, (_, readback), channels) in [
            (&mut sums.tangent_normal, band.tangent, 3),
            (&mut sums.object_normal, band.object, 3),
            (&mut sums.ambient_occlusion, band.ao, 1),
        ] {
            if let Some(map) = map {
                let values: &[f32] = bytemuck::cast_slice(read(readback)?);
                map.data[start * channels..(start + band_texels) * channels]
                    .copy_from_slice(&values[..band_texels * channels]);
            }
        }
        for (counts, (_, readback)) in [
            (&mut sums.hits, band.hits),
            (&mut sums.covered, band.covered),
        ] {
            let values: &[u32] = bytemuck::cast_slice(read(readback)?);
            for (dst, &src) in counts[start..start + band_texels].iter_mut().zip(values) {
                *dst = src.min(u16::MAX as u32) as u16;
            }
        }
        first_row = rows.end;
    }
    Ok(polyflow_bake::finish(sums, settings))
}

fn create_pipeline(res: &mut Resources<'_>, ray_query: bool) -> Result<(), RenderError> {
    let device = &res.ctx.device;
    let (spv, entry) = if ray_query {
        (BAKE_RT_SPV, c"bakeRtMain")
    } else {
        (BAKE_SPV, c"bakeMain")
    };
    let code = ash::util::read_spv(&mut Cursor::new(spv))
        .map_err(|e| RenderError::Other(e.to_string()))?;
    // SAFETY: valid SPIR-V produced by slangc.
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
    };
    let push = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(size_of::<BakePushConstants>() as u32)];
    let set_layouts = [res.set_layout];
    let mut layout_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
    if ray_query {
        layout_info = layout_info.set_layouts(&set_layouts);
    }
    // SAFETY: valid create info; destroyed with `res`.
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None) };
    let result = layout.map_err(RenderError::from).and_then(|layout| {
        res.layout = layout;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        // SAFETY: valid create info.
        let pipelines =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) }
                .map_err(|(_, e)| e)?;
        res.pipeline = pipelines[0];
        Ok(())
    });
    // SAFETY: modules may be destroyed once pipelines are created.
    unsafe { device.destroy_shader_module(module, None) };
    result
}

/// The descriptor set holding the scene's acceleration structure.
fn create_descriptor_set(res: &mut Resources<'_>) -> Result<vk::DescriptorSet, RenderError> {
    let device = &res.ctx.device;
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];
    // SAFETY: valid create infos; destroyed with `res`.
    unsafe {
        res.set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .descriptor_count(1)];
        res.descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&sizes),
            None,
        )?;
        let layouts = [res.set_layout];
        Ok(device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(res.descriptor_pool)
                .set_layouts(&layouts),
        )?[0])
    }
}

fn write_scene(res: &Resources<'_>, set: vk::DescriptorSet, scene: vk::AccelerationStructureKHR) {
    let structures = [scene];
    let mut accel_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
        .acceleration_structures(&structures);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .descriptor_count(1)
        .push_next(&mut accel_write);
    // SAFETY: the set and the acceleration structure are alive and unused.
    unsafe { res.ctx.device.update_descriptor_sets(&[write], &[]) };
}

/// Builds the acceleration structures for the reference mesh (whose
/// positions and indices `params` points at) and returns the top level.
fn build_scene(
    res: &mut Resources<'_>,
    params: &BakeParams,
    mesh: &polyflow_core::mesh::TriMesh,
) -> Result<vk::AccelerationStructureKHR, RenderError> {
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: params.ref_positions,
        })
        // Positions are uploaded as float4.
        .vertex_stride(16)
        .max_vertex(mesh.vertex_count().saturating_sub(1) as u32)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: params.ref_indices,
        });
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE);
    let (_, bottom) = build_acceleration(
        res,
        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        geometry,
        mesh.triangle_count() as u32,
    )?;

    // One instance of it, untransformed, culling nothing.
    let instance = vk::AccelerationStructureInstanceKHR {
        transform: vk::TransformMatrixKHR {
            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
        instance_custom_index_and_mask: vk::Packed24_8::new(0, 0xff),
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
            0,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
        ),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: bottom,
        },
    };
    // SAFETY: the instance is plain `repr(C)` data; we read its bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (&instance as *const vk::AccelerationStructureInstanceKHR).cast::<u8>(),
            size_of::<vk::AccelerationStructureInstanceKHR>(),
        )
    };
    let instances = res.upload_with(
        bytes,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        "bake instance",
    )?;
    let geometry = vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR {
            instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                .array_of_pointers(false)
                .data(vk::DeviceOrHostAddressConstKHR {
                    device_address: instances,
                }),
        });
    let (top, _) = build_acceleration(
        res,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
        geometry,
        1,
    )?;
    Ok(top)
}

/// Builds one acceleration structure on the GPU; returns it and its address.
fn build_acceleration(
    res: &mut Resources<'_>,
    ty: vk::AccelerationStructureTypeKHR,
    geometry: vk::AccelerationStructureGeometryKHR<'_>,
    primitives: u32,
) -> Result<(vk::AccelerationStructureKHR, u64), RenderError> {
    use vk::BufferUsageFlags as U;
    let ctx = res.ctx;
    let accel = res.accel.clone().expect("ray query path");
    let geometries = [geometry];
    let build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(ty)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    // SAFETY: valid build info; plain size query.
    unsafe {
        accel.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &build,
            &[primitives],
            &mut sizes,
        )
    };
    let storage = res.buffer(
        sizes.acceleration_structure_size as usize,
        U::ACCELERATION_STRUCTURE_STORAGE_KHR | U::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        "bake acceleration structure",
    )?;
    let create = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(res.buffers[storage].buffer)
        .size(sizes.acceleration_structure_size)
        .ty(ty);
    // SAFETY: valid create info on a live buffer; destroyed with `res`.
    let structure = unsafe { accel.create_acceleration_structure(&create, None)? };
    res.structures.push(structure);

    let alignment = scratch_alignment(ctx);
    let scratch = res.buffer(
        (sizes.build_scratch_size + alignment) as usize,
        U::STORAGE_BUFFER | U::SHADER_DEVICE_ADDRESS,
        MemoryLocation::GpuOnly,
        "bake scratch",
    )?;
    let scratch_address = res.buffers[scratch]
        .device_address(ctx)
        .next_multiple_of(alignment);
    let build =
        build
            .dst_acceleration_structure(structure)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch_address,
            });
    let range = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(primitives)];
    submit_and_wait(ctx, res.pool, |cmd| {
        // SAFETY: `cmd` is recording; every buffer the build reads or writes
        // is alive and large enough.
        unsafe { accel.cmd_build_acceleration_structures(cmd, &[build], &[&range]) };
    })?;
    res.free(scratch);
    // SAFETY: the structure was just built.
    let address = unsafe {
        accel.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                .acceleration_structure(structure),
        )
    };
    Ok((structure, address))
}

/// Alignment of acceleration structure build scratch memory.
fn scratch_alignment(ctx: &GpuContext) -> u64 {
    let mut accel = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut accel);
    // SAFETY: plain query on the context's physical device.
    unsafe {
        ctx.instance
            .get_physical_device_properties2(ctx.physical_device, &mut properties)
    };
    u64::from(accel.min_acceleration_structure_scratch_offset_alignment).max(1)
}

fn band_buffers(
    res: &mut Resources<'_>,
    texels: usize,
    maps: polyflow_bake::MapSet,
) -> Result<BandBuffers, RenderError> {
    use vk::BufferUsageFlags as U;
    let gpu_usage =
        U::STORAGE_BUFFER | U::SHADER_DEVICE_ADDRESS | U::TRANSFER_SRC | U::TRANSFER_DST;
    let mut pair = |bytes: usize, name: &str| -> Result<(usize, usize), RenderError> {
        Ok((
            res.buffer(bytes, gpu_usage, MemoryLocation::GpuOnly, name)?,
            res.buffer(bytes, U::TRANSFER_DST, MemoryLocation::GpuToCpu, name)?,
        ))
    };
    let f = size_of::<f32>();
    let tangent = pair(
        if maps.tangent_normal {
            texels * 3 * f
        } else {
            4
        },
        "bake tangent sums",
    )?;
    let object = pair(
        if maps.object_normal {
            texels * 3 * f
        } else {
            4
        },
        "bake object sums",
    )?;
    let ao = pair(
        if maps.ambient_occlusion {
            texels * f
        } else {
            4
        },
        "bake ao sums",
    )?;
    let hits = pair(texels * 4, "bake hits")?;
    let covered = pair(texels * 4, "bake covered")?;
    let host = U::STORAGE_BUFFER | U::SHADER_DEVICE_ADDRESS;
    Ok(BandBuffers {
        tangent,
        object,
        ao,
        hits,
        covered,
        raster_triangles: res.buffer(texels * 4, host, MemoryLocation::CpuToGpu, "bake raster")?,
        raster_bary: res.buffer(
            texels * 8,
            host,
            MemoryLocation::CpuToGpu,
            "bake barycentrics",
        )?,
        params: res.buffer(
            size_of::<BakeParams>(),
            host,
            MemoryLocation::CpuToGpu,
            "bake params",
        )?,
    })
}

/// Copies `data` into the start of a host-visible buffer.
fn write(res: &mut Resources<'_>, index: usize, data: &[u8]) -> Result<(), RenderError> {
    let mapped = res.buffers[index]
        .mapped_mut()
        .ok_or_else(|| RenderError::Other("upload buffer is not host visible".into()))?;
    mapped[..data.len()].copy_from_slice(data);
    Ok(())
}

/// Makes all memory writes of `src` stages visible to `dst` stages.
///
/// # Safety
/// `cmd` must be recording.
unsafe fn barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: vk::PipelineStageFlags2,
    dst: vk::PipelineStageFlags2,
) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(src)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(dst)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)];
    // SAFETY: the caller guarantees `cmd` is recording.
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(&barrier),
        )
    };
}

#[cfg(test)]
mod tests {
    use polyflow_bake::{BakeControl, Coverage, Map};
    use polyflow_core::glam::{Vec2, Vec3};
    use polyflow_core::halfedge::HalfEdgeMesh;
    use polyflow_core::mesh::TriMesh;
    use polyflow_spatial::Bvh;

    use super::*;

    /// A coarse torus with UVs (seams both ways) and a fine, bumpy one
    /// around it.
    fn scene() -> (HalfEdgeMesh, TriMesh) {
        use std::f32::consts::TAU;
        let (rings, sides) = (24, 12);
        let mut low = HalfEdgeMesh::new();
        let mut ids = Vec::new();
        for i in 0..rings {
            for j in 0..sides {
                let (a, b) = (i as f32 / rings as f32 * TAU, j as f32 / sides as f32 * TAU);
                // Around Y, like `primitives::torus`.
                let r = 1.0 + 0.35 * b.cos();
                ids.push(low.add_vertex(Vec3::new(r * a.cos(), 0.35 * b.sin(), r * a.sin())));
            }
        }
        let id = |i: usize, j: usize| ids[(i % rings) * sides + j % sides];
        let uv = |i: usize, j: usize| Vec2::new(i as f32 / rings as f32, j as f32 / sides as f32);
        for i in 0..rings {
            for j in 0..sides {
                // Facing out, as `primitives::torus` winds its triangles.
                let f = low
                    .add_face(&[id(i, j), id(i, j + 1), id(i + 1, j + 1), id(i + 1, j)])
                    .unwrap();
                low.set_face_uvs(f, &[uv(i, j), uv(i, j + 1), uv(i + 1, j + 1), uv(i + 1, j)]);
            }
        }
        let mut high = polyflow_core::primitives::torus(1.0, 0.36, 360, 180);
        for p in &mut high.positions {
            let bump = 0.015 * (p.x * 23.0).sin() * (p.y * 19.0).cos() * (p.z * 29.0).sin();
            *p += *p * bump;
        }
        high.compute_normals();
        (low, high)
    }

    fn angle(a: &[f32], b: &[f32]) -> f32 {
        let decode = |c: &[f32]| (Vec3::from_slice(c) * 2.0 - 1.0).normalize();
        decode(a)
            .dot(decode(b))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }

    #[test]
    fn gpu_matches_cpu() {
        // Shows Vulkan validation messages (POLYFLOW_VALIDATION=1).
        let _ = env_logger::builder().is_test(true).try_init();
        let ctx = match GpuContext::headless() {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("skipping: no GPU ({e})");
                return;
            }
        };
        let (retopo, high) = scene();
        let bvh = Bvh::build(&high);
        let reference = Reference {
            mesh: &high,
            bvh: &bvh,
        };
        let mut tracers = vec![Tracer::Compute];
        if ctx.ray_query {
            tracers.push(Tracer::RayQuery);
        }
        // Hard edges: at the UV seams, and around the inside of the ring.
        let mut hard = retopo.clone();
        let inner: Vec<_> = hard
            .edges()
            .filter(|&e| {
                let h = e.halfedge(0);
                let r = |v| {
                    let p = hard.position(v);
                    p.x.hypot(p.z)
                };
                r(hard.origin(h)) < 0.66 && r(hard.target(h)) < 0.66
            })
            .collect();
        assert!(!inner.is_empty());
        for e in inner {
            hard.set_hard(e, true);
        }
        let seams = polyflow_bake::NormalOptions {
            hard_uv_seams: true,
        };
        let cases = [(false, false), (true, false), (false, true)];
        for (tracer, (cage, split)) in tracers.iter().flat_map(|&t| cases.map(|c| (t, c))) {
            eprintln!("{tracer:?}, cage {cage}, hard edges {split}:");
            let cage = cage.then(|| polyflow_bake::Cage::automatic(&retopo, 0.05));
            let low = if split {
                BakeMesh::new(&hard, None, seams)
            } else {
                BakeMesh::with_cage(&retopo, cage.as_ref())
            };
            let mut settings = BakeSettings::for_scene(160, 96, 1.4);
            settings.samples = 2;
            settings.ao_samples = 16;
            let cpu =
                polyflow_bake::bake(&low, reference, &settings, &BakeControl::default()).unwrap();
            let control = BakeControl::default();
            let gpu = bake_with(&ctx, &low, reference, &settings, &control, tracer).unwrap();
            assert_eq!(control.rows_done(), settings.progress_rows());

            let texels = cpu.coverage.len();
            let differ = cpu
                .coverage
                .iter()
                .zip(&gpu.coverage)
                .filter(|(a, b)| a != b)
                .count();
            assert!(
                differ * 1000 <= texels,
                "coverage differs at {differ} texels"
            );
            assert!(
                cpu.hits > texels / 2,
                "the scene is mostly hit: {} hits, {} misses of {texels}",
                cpu.hits,
                cpu.misses
            );
            let hit =
                |i: usize| cpu.coverage[i] == Coverage::Hit && gpu.coverage[i] == Coverage::Hit;
            for (name, a, b) in [
                ("tangent", &cpu.tangent_normal, &gpu.tangent_normal),
                ("object", &cpu.object_normal, &gpu.object_normal),
            ] {
                let (a, b): (&Map, &Map) = (a.as_ref().unwrap(), b.as_ref().unwrap());
                let errors: Vec<f32> = (0..texels)
                    .filter(|&i| hit(i))
                    .map(|i| angle(&a.data[i * 3..i * 3 + 3], &b.data[i * 3..i * 3 + 3]))
                    .collect();
                let mean = errors.iter().sum::<f32>() / errors.len() as f32;
                let worst = errors.iter().filter(|&&e| e > 0.5).count();
                let max = errors.iter().fold(0.0f32, |m, &e| m.max(e));
                eprintln!("{name}: mean {mean:.5} deg, max {max:.3} deg, {worst} over 0.5 deg");
                assert!(mean < 0.01, "{name}: mean {mean} deg");
                assert!(
                    worst * 1000 <= errors.len(),
                    "{name}: {worst} texels over 0.5 deg"
                );
            }
            let (a, b) = (
                cpu.ambient_occlusion.unwrap(),
                gpu.ambient_occlusion.unwrap(),
            );
            let diffs: Vec<f32> = (0..texels)
                .filter(|&i| hit(i))
                .map(|i| (a.data[i] - b.data[i]).abs())
                .collect();
            let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
            eprintln!("ao: mean difference {mean:.6}; coverage differs at {differ} of {texels}");
            assert!(mean < 0.002, "ao: mean difference {mean}");
        }
    }

    #[test]
    fn layouts_match_the_shader() {
        assert_eq!(size_of::<BakeParams>(), 14 * 8 + 8 * 4);
        assert_eq!(size_of::<BakePushConstants>(), 16);
        assert_eq!(polyflow_spatial::STACK_SIZE, 88, "STACK_SIZE in bake.slang");
    }
}
