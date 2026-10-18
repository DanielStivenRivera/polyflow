//! GPU buffers and images backed by `gpu-allocator`.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::RenderError;
use crate::context::GpuContext;

/// A buffer and its memory. Must be destroyed explicitly with [`GpuBuffer::destroy`].
#[derive(Debug)]
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    allocation: Option<Allocation>,
}

impl GpuBuffer {
    pub fn new(
        ctx: &GpuContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self, RenderError> {
        let info = vk::BufferCreateInfo::default()
            .size(size.max(1))
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid device and create info.
        let buffer = unsafe { ctx.device.create_buffer(&info, None)? };
        // SAFETY: `buffer` was just created from this device.
        let requirements = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        let allocation = match ctx.allocator().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: the buffer is unused.
                unsafe { ctx.device.destroy_buffer(buffer, None) };
                return Err(e.into());
            }
        };
        // SAFETY: the allocation satisfies the buffer's requirements and is not bound elsewhere.
        unsafe {
            ctx.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?
        };
        Ok(Self {
            buffer,
            allocation: Some(allocation),
        })
    }

    /// Creates a device-local buffer filled with `data`, via a staging buffer.
    /// Blocks until the copy has finished.
    pub fn new_device_local(
        ctx: &GpuContext,
        command_pool: vk::CommandPool,
        data: &[u8],
        usage: vk::BufferUsageFlags,
        name: &str,
    ) -> Result<Self, RenderError> {
        let size = data.len() as vk::DeviceSize;
        let mut staging = Self::new(
            ctx,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "staging",
        )?;
        let result = (|| {
            staging.mapped_mut().ok_or(RenderError::Other(
                "staging buffer is not host visible".into(),
            ))?[..data.len()]
                .copy_from_slice(data);
            let dst = Self::new(
                ctx,
                size,
                usage | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                name,
            )?;
            let copy = submit_and_wait(ctx, command_pool, |cmd| {
                let region = vk::BufferCopy::default().size(size);
                // SAFETY: both buffers are alive and large enough; `cmd` is recording.
                unsafe {
                    ctx.device
                        .cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[region])
                };
            });
            match copy {
                Ok(()) => Ok(dst),
                Err(e) => {
                    let mut dst = dst;
                    dst.destroy(ctx);
                    Err(e)
                }
            }
        })();
        staging.destroy(ctx);
        result
    }

    pub fn mapped_mut(&mut self) -> Option<&mut [u8]> {
        self.allocation.as_mut().and_then(|a| a.mapped_slice_mut())
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        if let Some(allocation) = self.allocation.take() {
            // SAFETY: the caller guarantees the GPU no longer uses the buffer.
            unsafe { ctx.device.destroy_buffer(self.buffer, None) };
            if let Err(e) = ctx.allocator().free(allocation) {
                log::error!("failed to free buffer memory: {e}");
            }
        }
    }
}

/// A 2D image, its default view and its memory. Must be destroyed explicitly.
#[derive(Debug)]
pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    allocation: Option<Allocation>,
}

impl GpuImage {
    pub fn new(
        ctx: &GpuContext,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
        name: &str,
    ) -> Result<Self, RenderError> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: valid device and create info.
        let image = unsafe { ctx.device.create_image(&info, None)? };
        // SAFETY: `image` was just created from this device.
        let requirements = unsafe { ctx.device.get_image_memory_requirements(image) };
        let allocation = ctx.allocator().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        let allocation = match allocation {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: the image is unused.
                unsafe { ctx.device.destroy_image(image, None) };
                return Err(e.into());
            }
        };
        let mut result = Self {
            image,
            view: vk::ImageView::null(),
            format,
            allocation: Some(allocation),
        };
        let alloc = result.allocation.as_ref().unwrap();
        // SAFETY: the allocation satisfies the image's requirements.
        let bound = unsafe {
            ctx.device
                .bind_image_memory(image, alloc.memory(), alloc.offset())
        };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range(aspect));
        // SAFETY: the image is bound to memory (checked first).
        match bound.and_then(|()| unsafe { ctx.device.create_image_view(&view_info, None) }) {
            Ok(view) => {
                result.view = view;
                Ok(result)
            }
            Err(e) => {
                result.destroy(ctx);
                Err(e.into())
            }
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        if let Some(allocation) = self.allocation.take() {
            // SAFETY: the caller guarantees the GPU no longer uses the image.
            unsafe {
                if self.view != vk::ImageView::null() {
                    ctx.device.destroy_image_view(self.view, None);
                }
                ctx.device.destroy_image(self.image, None);
            }
            if let Err(e) = ctx.allocator().free(allocation) {
                log::error!("failed to free image memory: {e}");
            }
        }
    }
}

pub fn subresource_range(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect)
        .level_count(1)
        .layer_count(1)
}

/// Records commands with `record` into a one-time command buffer, submits it
/// and blocks until it completes.
pub fn submit_and_wait(
    ctx: &GpuContext,
    command_pool: vk::CommandPool,
    record: impl FnOnce(vk::CommandBuffer),
) -> Result<(), RenderError> {
    let device = &ctx.device;
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: the pool belongs to this device and is only used from this thread.
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];
    // SAFETY: `cmd` and `fence` are created here and released before returning,
    // after the fence confirms the GPU is done with them.
    let result = unsafe {
        (|| {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            record(cmd);
            device.end_command_buffer(cmd)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&cmds);
            let waited = device
                .queue_submit2(ctx.queue, &[submit], fence)
                .and_then(|()| device.wait_for_fences(&[fence], true, u64::MAX));
            device.destroy_fence(fence, None);
            waited
        })()
    };
    // SAFETY: the command buffer has finished executing (or was never submitted).
    unsafe { device.free_command_buffers(command_pool, &[cmd]) };
    Ok(result?)
}
