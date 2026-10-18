//! Swapchain creation and recreation.

use ash::khr::swapchain;
use ash::vk;

use crate::RenderError;
use crate::context::GpuContext;
use crate::memory::subresource_range;

pub struct Swapchain {
    loader: swapchain::Device,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
    /// One per image: a present may still be waiting on the semaphore of an
    /// image when the next frame starts, so they cannot be per frame in flight.
    pub render_finished: Vec<vk::Semaphore>,
}

impl std::fmt::Debug for Swapchain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Swapchain")
            .field("format", &self.format)
            .field("extent", &self.extent)
            .field("present_mode", &self.present_mode)
            .field("images", &self.images.len())
            .finish_non_exhaustive()
    }
}

impl Swapchain {
    /// Creates a swapchain for the context's surface. `old` (possibly null) is
    /// handed to the driver for resource reuse; the caller still destroys it.
    pub fn new(
        ctx: &GpuContext,
        window_extent: vk::Extent2D,
        vsync: bool,
        old: vk::SwapchainKHR,
    ) -> Result<Self, RenderError> {
        let (pd, surface) = (ctx.physical_device, ctx.surface);
        // SAFETY: plain queries on a valid physical device and surface.
        let (caps, formats, modes) = unsafe {
            (
                ctx.surface_loader
                    .get_physical_device_surface_capabilities(pd, surface)?,
                ctx.surface_loader
                    .get_physical_device_surface_formats(pd, surface)?,
                ctx.surface_loader
                    .get_physical_device_surface_present_modes(pd, surface)?,
            )
        };

        let surface_format = choose_surface_format(&formats)
            .ok_or(RenderError::Other("surface has no formats".into()))?;
        let present_mode = choose_present_mode(&modes, vsync);
        let extent = surface_extent(&caps, window_extent);
        if extent.width == 0 || extent.height == 0 {
            return Err(RenderError::Other(
                "cannot create a swapchain for a zero-sized surface".into(),
            ));
        }
        let mut image_count = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            image_count = image_count.min(caps.max_image_count);
        }
        let composite_alpha = [
            vk::CompositeAlphaFlagsKHR::OPAQUE,
            vk::CompositeAlphaFlagsKHR::INHERIT,
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
            vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        ]
        .into_iter()
        .find(|&a| caps.supported_composite_alpha.contains(a))
        .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old);

        let loader = swapchain::Device::new(&ctx.instance, &ctx.device);
        // SAFETY: valid create info for this device's surface.
        let handle = unsafe { loader.create_swapchain(&info, None)? };
        let mut sc = Self {
            loader,
            handle,
            images: vec![],
            views: vec![],
            format: surface_format.format,
            extent,
            present_mode,
            render_finished: vec![],
        };
        if let Err(e) = sc.create_image_resources(ctx) {
            sc.destroy(ctx);
            return Err(e);
        }
        log::debug!("Created {sc:?}");
        Ok(sc)
    }

    fn create_image_resources(&mut self, ctx: &GpuContext) -> Result<(), RenderError> {
        // SAFETY: `handle` is a live swapchain from this loader.
        self.images = unsafe { self.loader.get_swapchain_images(self.handle)? };
        for &image in &self.images {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.format)
                .subresource_range(subresource_range(vk::ImageAspectFlags::COLOR));
            // SAFETY: `image` belongs to the swapchain; the objects are tracked
            // in `self` and released in `destroy`.
            unsafe {
                self.views
                    .push(ctx.device.create_image_view(&view_info, None)?);
                self.render_finished
                    .push(ctx.device.create_semaphore(&Default::default(), None)?);
            }
        }
        Ok(())
    }

    /// Size a swapchain created now would have; zero while the window is
    /// minimized on some platforms.
    pub fn current_surface_extent(
        ctx: &GpuContext,
        window_extent: vk::Extent2D,
    ) -> Result<vk::Extent2D, RenderError> {
        // SAFETY: plain query on a valid physical device and surface.
        let caps = unsafe {
            ctx.surface_loader
                .get_physical_device_surface_capabilities(ctx.physical_device, ctx.surface)?
        };
        Ok(surface_extent(&caps, window_extent))
    }

    /// Acquires the next image; `Ok((index, suboptimal))`.
    pub fn acquire(&self, signal: vk::Semaphore) -> Result<(u32, bool), vk::Result> {
        // SAFETY: live swapchain; the semaphore is unsignaled with no pending operations.
        unsafe {
            self.loader
                .acquire_next_image(self.handle, u64::MAX, signal, vk::Fence::null())
        }
    }

    /// Presents `image`; `Ok(suboptimal)`.
    pub fn present(&self, queue: vk::Queue, image: u32) -> Result<bool, vk::Result> {
        let wait = [self.render_finished[image as usize]];
        let swapchains = [self.handle];
        let indices = [image];
        let info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: `image` was acquired and its rendering signals `wait`.
        unsafe { self.loader.queue_present(queue, &info) }
    }

    /// Destroys the swapchain. The caller guarantees the GPU is idle.
    pub fn destroy(&mut self, ctx: &GpuContext) {
        // SAFETY: the device is idle, so none of these objects are in use.
        unsafe {
            for &s in &self.render_finished {
                ctx.device.destroy_semaphore(s, None);
            }
            for &v in &self.views {
                ctx.device.destroy_image_view(v, None);
            }
            if self.handle != vk::SwapchainKHR::null() {
                self.loader.destroy_swapchain(self.handle, None);
            }
        }
        self.render_finished.clear();
        self.views.clear();
        self.images.clear();
        self.handle = vk::SwapchainKHR::null();
    }
}

fn surface_extent(caps: &vk::SurfaceCapabilitiesKHR, window_extent: vk::Extent2D) -> vk::Extent2D {
    if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: window_extent
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: window_extent
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    }
}

fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    let preferred = [vk::Format::B8G8R8A8_SRGB, vk::Format::R8G8B8A8_SRGB];
    formats
        .iter()
        .find(|f| {
            preferred.contains(&f.format) && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .copied()
}

fn choose_present_mode(modes: &[vk::PresentModeKHR], vsync: bool) -> vk::PresentModeKHR {
    if vsync {
        return vk::PresentModeKHR::FIFO;
    }
    [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::IMMEDIATE]
        .into_iter()
        .find(|m| modes.contains(m))
        .unwrap_or(vk::PresentModeKHR::FIFO)
}
