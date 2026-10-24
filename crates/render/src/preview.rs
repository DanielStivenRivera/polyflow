//! Preview of a bake: the low-poly as baked, lit with its normal map and
//! darkened by its occlusion, so the result can be judged without leaving
//! Polyflow. The shading follows MikkTSpace (unnormalized interpolated
//! normal and tangent, bitangent per pixel), as the baker assumes.

use std::io::Cursor;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use polyflow_core::glam::{Mat4, Vec3, Vec4};

use crate::RenderError;
use crate::context::GpuContext;
use crate::memory::{GpuBuffer, submit_and_wait};

const PREVIEW_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/preview.spv"));

/// What the preview shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreviewMode {
    /// Lit with the normal map, darkened by the occlusion.
    Lit,
    /// The tangent-space normal map's colors.
    NormalMap,
    /// The occlusion map.
    Occlusion,
}

/// One corner of a preview triangle. Must match `preview.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, PartialEq)]
pub struct PreviewVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// MikkTSpace tangent, bitangent sign last.
    pub tangent: [f32; 4],
    pub uv: [f32; 2],
}

/// A baked map for the preview: `channels` floats per texel, rows from the
/// top, values 0..1.
#[derive(Debug, Clone, Copy)]
pub struct PreviewMap<'a> {
    pub width: u32,
    pub height: u32,
    pub channels: usize,
    pub data: &'a [f32],
}

/// What to preview.
#[derive(Debug, Clone, Copy)]
pub struct PreviewData<'a> {
    /// Three per triangle.
    pub vertices: &'a [PreviewVertex],
    pub normal_map: Option<PreviewMap<'a>>,
    /// Green down in the normal map.
    pub directx: bool,
    pub occlusion: Option<PreviewMap<'a>>,
}

/// Must match `PushConstants` in `shaders/preview.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PreviewPushConstants {
    view_proj: Mat4,
    eye: Vec4,
    mode: u32,
    directx: u32,
    pad: [u32; 2],
}

/// A sampled texture with a full mip chain.
#[derive(Debug)]
struct Texture {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl Texture {
    fn destroy(&mut self, ctx: &GpuContext) {
        if let Some(allocation) = self.allocation.take() {
            // SAFETY: the caller guarantees the GPU no longer uses the image.
            unsafe {
                ctx.device.destroy_image_view(self.view, None);
                ctx.device.destroy_image(self.image, None);
            }
            if let Err(e) = ctx.allocator().free(allocation) {
                log::error!("failed to free texture memory: {e}");
            }
        }
    }
}

/// GPU resources of the bake being previewed.
#[derive(Debug)]
struct Uploaded {
    vertices: GpuBuffer,
    vertex_count: u32,
    normal_map: Texture,
    occlusion: Texture,
    directx: bool,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

/// Draws bake previews.
#[derive(Debug)]
pub struct PreviewRenderer {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    uploaded: Option<Uploaded>,
}

impl PreviewRenderer {
    pub fn new(
        ctx: &GpuContext,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Result<Self, RenderError> {
        let mut renderer = Self {
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            sampler: vk::Sampler::null(),
            uploaded: None,
        };
        match renderer.create(ctx, color_format, depth_format) {
            Ok(()) => Ok(renderer),
            Err(e) => {
                renderer.destroy(ctx);
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
        let bindings = [0, 1].map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        });
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(vk::LOD_CLAMP_NONE);
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<PreviewPushConstants>() as u32)];
        // SAFETY: valid create infos; everything is destroyed in `destroy`.
        unsafe {
            self.set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            self.sampler = device.create_sampler(&sampler_info, None)?;
            let set_layouts = [self.set_layout];
            self.layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )?;
        }

        let code = ash::util::read_spv(&mut Cursor::new(PREVIEW_SPV))
            .map_err(|e| RenderError::Other(e.to_string()))?;
        // SAFETY: valid SPIR-V produced by slangc.
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(module)
                .name(c"previewVertex"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(module)
                .name(c"previewFragment"),
        ];
        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<PreviewVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];
        let attr = |location, format, offset: usize| {
            vk::VertexInputAttributeDescription::default()
                .location(location)
                .binding(0)
                .format(format)
                .offset(offset as u32)
        };
        let attributes = [
            attr(
                0,
                vk::Format::R32G32B32_SFLOAT,
                std::mem::offset_of!(PreviewVertex, position),
            ),
            attr(
                1,
                vk::Format::R32G32B32_SFLOAT,
                std::mem::offset_of!(PreviewVertex, normal),
            ),
            attr(
                2,
                vk::Format::R32G32B32A32_SFLOAT,
                std::mem::offset_of!(PreviewVertex, tangent),
            ),
            attr(
                3,
                vk::Format::R32G32_SFLOAT,
                std::mem::offset_of!(PreviewVertex, uv),
            ),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
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
        // Reverse-Z, like the scene.
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

    /// Replaces the bake being previewed (`None` to drop it). Waits for the
    /// GPU to be idle.
    pub fn set(
        &mut self,
        ctx: &GpuContext,
        command_pool: vk::CommandPool,
        data: Option<&PreviewData<'_>>,
    ) -> Result<(), RenderError> {
        // SAFETY: waiting for idle so the old resources are no longer in use.
        unsafe { ctx.device.device_wait_idle()? };
        if let Some(old) = self.uploaded.take() {
            destroy_uploaded(ctx, old);
        }
        let Some(data) = data else { return Ok(()) };
        if data.vertices.is_empty() {
            return Ok(());
        }
        let vertices = GpuBuffer::new_device_local(
            ctx,
            command_pool,
            bytemuck::cast_slice(data.vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "preview vertices",
        )?;
        let mut uploaded = Uploaded {
            vertices,
            vertex_count: data.vertices.len() as u32,
            normal_map: Texture {
                image: vk::Image::null(),
                view: vk::ImageView::null(),
                allocation: None,
            },
            occlusion: Texture {
                image: vk::Image::null(),
                view: vk::ImageView::null(),
                allocation: None,
            },
            directx: data.directx,
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
        };
        let result = (|| {
            // Missing maps become a single neutral texel.
            let flat = [0.5, 0.5, 1.0];
            let normal = data.normal_map.unwrap_or(PreviewMap {
                width: 1,
                height: 1,
                channels: 3,
                data: &flat,
            });
            uploaded.normal_map = upload_texture(ctx, command_pool, &normal, "preview normal map")?;
            let white = [1.0];
            let occlusion = data.occlusion.unwrap_or(PreviewMap {
                width: 1,
                height: 1,
                channels: 1,
                data: &white,
            });
            uploaded.occlusion =
                upload_texture(ctx, command_pool, &occlusion, "preview occlusion")?;
            self.write_descriptors(ctx, &mut uploaded)
        })();
        match result {
            Ok(()) => {
                self.uploaded = Some(uploaded);
                Ok(())
            }
            Err(e) => {
                destroy_uploaded(ctx, uploaded);
                Err(e)
            }
        }
    }

    fn write_descriptors(
        &self,
        ctx: &GpuContext,
        uploaded: &mut Uploaded,
    ) -> Result<(), RenderError> {
        let device = &ctx.device;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(2)];
        let layouts = [self.set_layout];
        // SAFETY: valid create infos; the pool is destroyed with `uploaded`.
        unsafe {
            uploaded.pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            uploaded.set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(uploaded.pool)
                    .set_layouts(&layouts),
            )?[0];
        }
        let images = [&uploaded.normal_map, &uploaded.occlusion].map(|t| {
            [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(t.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        });
        let writes: Vec<vk::WriteDescriptorSet<'_>> = images
            .iter()
            .enumerate()
            .map(|(binding, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(uploaded.set)
                    .dst_binding(binding as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            })
            .collect();
        // SAFETY: the set, views and sampler are alive and unused.
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(())
    }

    /// Records the preview draw, inside the scene's rendering scope with the
    /// viewport and scissor set.
    pub fn draw(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        view_proj: Mat4,
        eye: Vec3,
        mode: PreviewMode,
    ) {
        let Some(uploaded) = &self.uploaded else {
            return;
        };
        let push = PreviewPushConstants {
            view_proj,
            eye: eye.extend(1.0),
            mode: match mode {
                PreviewMode::Lit => 0,
                PreviewMode::NormalMap => 1,
                PreviewMode::Occlusion => 2,
            },
            directx: u32::from(uploaded.directx),
            pad: [0; 2],
        };
        // SAFETY: `cmd` is recording in a compatible rendering scope; the
        // resources stay alive until `set` replaces them after waiting idle.
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[uploaded.set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &[uploaded.vertices.buffer], &[0]);
            device.cmd_draw(cmd, uploaded.vertex_count, 1, 0, 0);
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        if let Some(uploaded) = self.uploaded.take() {
            destroy_uploaded(ctx, uploaded);
        }
        // SAFETY: the caller guarantees the GPU is idle; null handles are ignored.
        unsafe {
            ctx.device.destroy_pipeline(self.pipeline, None);
            ctx.device.destroy_pipeline_layout(self.layout, None);
            ctx.device.destroy_sampler(self.sampler, None);
            ctx.device
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn destroy_uploaded(ctx: &GpuContext, mut uploaded: Uploaded) {
    uploaded.vertices.destroy(ctx);
    uploaded.normal_map.destroy(ctx);
    uploaded.occlusion.destroy(ctx);
    // SAFETY: the GPU is idle (callers wait); null handles are ignored.
    unsafe { ctx.device.destroy_descriptor_pool(uploaded.pool, None) };
}

/// 8-bit RGBA (or R) mip levels of `map`, largest first, each halving the
/// previous with a box filter.
fn mip_chain(map: &PreviewMap<'_>) -> (vk::Format, Vec<(u32, u32, Vec<u8>)>) {
    let (format, out_channels) = match map.channels {
        1 => (vk::Format::R8_UNORM, 1),
        _ => (vk::Format::R8G8B8A8_UNORM, 4),
    };
    let to_byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut levels = Vec::new();
    // Level 0 straight from the floats, alpha opaque.
    let mut current: Vec<f32> =
        Vec::with_capacity(map.width as usize * map.height as usize * out_channels);
    for texel in map.data.chunks(map.channels) {
        for c in 0..out_channels {
            current.push(match (out_channels, c) {
                (4, 3) => 1.0,
                _ => texel[c.min(map.channels - 1)],
            });
        }
    }
    let (mut w, mut h) = (map.width as usize, map.height as usize);
    loop {
        levels.push((
            w as u32,
            h as u32,
            current.iter().map(|&v| to_byte(v)).collect(),
        ));
        if w == 1 && h == 1 {
            break;
        }
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = vec![0.0; nw * nh * out_channels];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..out_channels {
                    let mut sum = 0.0;
                    let mut n = 0.0;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((2 * x + dx).min(w - 1), (2 * y + dy).min(h - 1));
                        sum += current[(sy * w + sx) * out_channels + c];
                        n += 1.0;
                    }
                    next[(y * nw + x) * out_channels + c] = sum / n;
                }
            }
        }
        current = next;
        (w, h) = (nw, nh);
    }
    (format, levels)
}

fn upload_texture(
    ctx: &GpuContext,
    command_pool: vk::CommandPool,
    map: &PreviewMap<'_>,
    name: &str,
) -> Result<Texture, RenderError> {
    let (format, levels) = mip_chain(map);
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: map.width,
            height: map.height,
            depth: 1,
        })
        .mip_levels(levels.len() as u32)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: valid device and create info.
    let image = unsafe { ctx.device.create_image(&info, None)? };
    // SAFETY: `image` was just created from this device.
    let requirements = unsafe { ctx.device.get_image_memory_requirements(image) };
    let allocation = match ctx.allocator().allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    }) {
        Ok(a) => a,
        Err(e) => {
            // SAFETY: the image is unused.
            unsafe { ctx.device.destroy_image(image, None) };
            return Err(e.into());
        }
    };
    let mut texture = Texture {
        image,
        view: vk::ImageView::null(),
        allocation: Some(allocation),
    };
    let result = (|| {
        let alloc = texture.allocation.as_ref().expect("just allocated");
        // SAFETY: the allocation satisfies the image's requirements.
        unsafe {
            ctx.device
                .bind_image_memory(image, alloc.memory(), alloc.offset())?
        };
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(levels.len() as u32)
            .layer_count(1);
        // All levels in one staging buffer.
        let bytes: Vec<u8> = levels
            .iter()
            .flat_map(|(_, _, b)| b.iter().copied())
            .collect();
        let mut staging = GpuBuffer::new(
            ctx,
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "preview staging",
        )?;
        let copied = (|| {
            staging
                .mapped_mut()
                .ok_or_else(|| RenderError::Other("staging buffer is not host visible".into()))?
                [..bytes.len()]
                .copy_from_slice(&bytes);
            let mut offset = 0u64;
            let regions: Vec<vk::BufferImageCopy> = levels
                .iter()
                .enumerate()
                .map(|(level, (w, h, b))| {
                    let region = vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(
                            vk::ImageSubresourceLayers::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .mip_level(level as u32)
                                .layer_count(1),
                        )
                        .image_extent(vk::Extent3D {
                            width: *w,
                            height: *h,
                            depth: 1,
                        });
                    offset += b.len() as u64;
                    region
                })
                .collect();
            submit_and_wait(ctx, command_pool, |cmd| {
                let to_transfer = [vk::ImageMemoryBarrier2::default()
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(image)
                    .subresource_range(range)];
                let to_shader = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(image)
                    .subresource_range(range)];
                // SAFETY: `cmd` is recording; the image and staging buffer are
                // alive and sized for the regions.
                unsafe {
                    ctx.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&to_transfer),
                    );
                    ctx.device.cmd_copy_buffer_to_image(
                        cmd,
                        staging.buffer,
                        image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    );
                    ctx.device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().image_memory_barriers(&to_shader),
                    );
                }
            })
        })();
        staging.destroy(ctx);
        copied?;
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(range);
        // SAFETY: the image is bound and filled.
        texture.view = unsafe { ctx.device.create_image_view(&view_info, None)? };
        Ok(())
    })();
    match result {
        Ok(()) => Ok(texture),
        Err(e) => {
            texture.destroy(ctx);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_match_the_shader() {
        assert_eq!(size_of::<PreviewVertex>(), 48);
        assert_eq!(size_of::<PreviewPushConstants>(), 96);
    }

    #[test]
    fn mips_halve_down_to_one_texel() {
        let data: Vec<f32> = (0..5 * 3).map(|i| (i % 2) as f32).collect();
        let map = PreviewMap {
            width: 5,
            height: 3,
            channels: 1,
            data: &data,
        };
        let (format, levels) = mip_chain(&map);
        assert_eq!(format, vk::Format::R8_UNORM);
        let sizes: Vec<(u32, u32)> = levels.iter().map(|(w, h, _)| (*w, *h)).collect();
        assert_eq!(sizes, [(5, 3), (2, 1), (1, 1)]);
        assert_eq!(levels[0].2.len(), 15);
        let rgb = [0.0, 0.5, 1.0];
        let (format, levels) = mip_chain(&PreviewMap {
            width: 1,
            height: 1,
            channels: 3,
            data: &rgb,
        });
        assert_eq!(format, vk::Format::R8G8B8A8_UNORM);
        assert_eq!(levels[0].2, [0, 128, 255, 255]);
    }
}
