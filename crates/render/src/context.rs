//! Vulkan instance, device and memory allocator.

use std::ffi::{CStr, c_void};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use ash::ext::debug_utils;
use ash::khr::{surface, swapchain};
use ash::{Device, Entry, Instance, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::RenderError;

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Owns the core Vulkan objects shared by the whole renderer.
///
/// Everything created from the device must be destroyed before this is dropped.
pub struct GpuContext {
    // Kept alive because it owns the loaded Vulkan library.
    _entry: Entry,
    pub instance: Instance,
    debug: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub surface_loader: surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub device: Device,
    pub queue_family: u32,
    /// Valid bits in timestamps written on `queue`; 0 if unsupported.
    pub timestamp_valid_bits: u32,
    pub queue: vk::Queue,
    pub allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &self.device_name())
            .finish_non_exhaustive()
    }
}

impl GpuContext {
    pub fn new(window: &(impl HasDisplayHandle + HasWindowHandle)) -> Result<Self, RenderError> {
        let display = window
            .display_handle()
            .map_err(|e| RenderError::Window(e.to_string()))?;
        let window_handle = window
            .window_handle()
            .map_err(|e| RenderError::Window(e.to_string()))?;

        // SAFETY: loading the system Vulkan library; the entry is kept alive in `Self`.
        let entry = unsafe { Entry::load()? };

        let validation = validation_requested() && layer_available(&entry, VALIDATION_LAYER);
        if validation_requested() && !validation {
            log::warn!(
                "Vulkan validation layers requested but not installed (install the Vulkan SDK)"
            );
        }

        let mut extensions = ash_window::enumerate_required_extensions(display.as_raw())?.to_vec();
        if validation {
            extensions.push(debug_utils::NAME.as_ptr());
        }
        let layers = if validation {
            vec![VALIDATION_LAYER.as_ptr()]
        } else {
            vec![]
        };

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Polyflow")
            .engine_name(c"Polyflow")
            .api_version(vk::API_VERSION_1_3);
        let mut debug_info = debug_messenger_info();
        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        if validation {
            // Also reports problems during instance creation and destruction.
            instance_info = instance_info.push_next(&mut debug_info);
        }
        // SAFETY: all pointers in `instance_info` outlive the call.
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let debug = if validation {
            let loader = debug_utils::Instance::new(&entry, &instance);
            // SAFETY: valid instance created with the debug utils extension enabled.
            let messenger =
                unsafe { loader.create_debug_utils_messenger(&debug_messenger_info(), None)? };
            log::info!("Vulkan validation layers enabled");
            Some((loader, messenger))
        } else {
            None
        };

        let surface_loader = surface::Instance::new(&entry, &instance);
        // SAFETY: the handles come from a live window that outlives the renderer.
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display.as_raw(),
                window_handle.as_raw(),
                None,
            )?
        };

        let (physical_device, queue_family) =
            pick_physical_device(&instance, &surface_loader, surface)?;
        // SAFETY: `physical_device` was enumerated from `instance`.
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let priorities = [1.0];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_extensions = [swapchain::NAME.as_ptr()];
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .draw_indirect_count(true);
        let core_features = vk::PhysicalDeviceFeatures::default()
            .multi_draw_indirect(true)
            .draw_indirect_first_instance(true);
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .features(core_features)
            .push_next(&mut features12)
            .push_next(&mut features13);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features);
        // SAFETY: the device supports these features and extensions (checked in `pick_physical_device`).
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        // SAFETY: one queue was requested from this family.
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        // SAFETY: plain query on an enumerated physical device.
        let timestamp_valid_bits =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                [queue_family as usize]
                .timestamp_valid_bits;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let ctx = Self {
            _entry: entry,
            instance,
            debug,
            surface_loader,
            surface,
            physical_device,
            properties,
            device,
            queue_family,
            timestamp_valid_bits,
            queue,
            allocator: ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
        };
        log::info!("Using GPU: {}", ctx.device_name());
        Ok(ctx)
    }

    pub fn device_name(&self) -> &str {
        self.properties
            .device_name_as_c_str()
            .ok()
            .and_then(|s| s.to_str().ok())
            .unwrap_or("unknown GPU")
    }

    pub fn allocator(&self) -> std::sync::MutexGuard<'_, Allocator> {
        self.allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // SAFETY: the renderer has destroyed every object created from the device;
        // the remaining objects are destroyed in reverse creation order.
        unsafe {
            let _ = self.device.device_wait_idle();
            // The allocator frees its memory blocks on drop, so it must go before the device.
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// Validation defaults to on in debug builds; `POLYFLOW_VALIDATION=0/1` overrides.
fn validation_requested() -> bool {
    match std::env::var("POLYFLOW_VALIDATION") {
        Ok(v) => v != "0",
        Err(_) => cfg!(debug_assertions),
    }
}

fn layer_available(entry: &Entry, name: &CStr) -> bool {
    // SAFETY: plain query on a loaded entry.
    let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
    layers.iter().any(|l| l.layer_name_as_c_str() == Ok(name))
}

fn debug_messenger_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the validation layer passes a valid callback data pointer for the
    // duration of the call.
    let message = unsafe { data.as_ref() }
        .and_then(|d| unsafe { d.message_as_c_str() })
        .map(|m| m.to_string_lossy())
        .unwrap_or_default();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!(target: "vulkan", "[{kind:?}] {message}");
    } else {
        log::warn!(target: "vulkan", "[{kind:?}] {message}");
    }
    vk::FALSE
}

/// Picks a Vulkan 1.3 GPU that can present to `surface`, preferring discrete
/// GPUs. `POLYFLOW_GPU=<substring>` forces a GPU by name.
fn pick_physical_device(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32), RenderError> {
    let forced = std::env::var("POLYFLOW_GPU").ok().map(|s| s.to_lowercase());
    // SAFETY: plain queries on a valid instance and surface.
    let devices = unsafe { instance.enumerate_physical_devices()? };

    let mut best: Option<(u32, vk::PhysicalDevice, u32)> = None;
    for pd in devices {
        // SAFETY: as above.
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if props.api_version < vk::API_VERSION_1_3 {
            log::debug!("Skipping {name}: Vulkan 1.3 not supported");
            continue;
        }
        if !supports_required_features(instance, pd) {
            log::debug!("Skipping {name}: missing required features");
            continue;
        }
        let Some(queue_family) = find_queue_family(instance, surface_loader, surface, pd) else {
            log::debug!("Skipping {name}: no graphics queue that can present");
            continue;
        };
        let mut score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 3,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            _ => 0,
        };
        if forced
            .as_ref()
            .is_some_and(|f| name.to_lowercase().contains(f.as_str()))
        {
            score += 100;
        }
        if best.is_none_or(|(s, ..)| score > s) {
            best = Some((score, pd, queue_family));
        }
    }
    best.map(|(_, pd, qf)| (pd, qf))
        .ok_or(RenderError::NoSuitableGpu)
}

fn supports_required_features(instance: &Instance, pd: vk::PhysicalDevice) -> bool {
    // SAFETY: plain queries on an enumerated physical device.
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    if !extensions
        .iter()
        .any(|e| e.extension_name_as_c_str() == Ok(swapchain::NAME))
    {
        return false;
    }
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut features12)
        .push_next(&mut features13);
    // SAFETY: as above.
    unsafe { instance.get_physical_device_features2(pd, &mut features) };
    let core = features.features;
    [
        features13.dynamic_rendering,
        features13.synchronization2,
        features12.buffer_device_address,
        features12.draw_indirect_count,
        core.multi_draw_indirect,
        core.draw_indirect_first_instance,
    ]
    .iter()
    .all(|&f| f == vk::TRUE)
}

fn find_queue_family(
    instance: &Instance,
    surface_loader: &surface::Instance,
    surface: vk::SurfaceKHR,
    pd: vk::PhysicalDevice,
) -> Option<u32> {
    // SAFETY: plain queries on an enumerated physical device and a valid surface.
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    families.iter().enumerate().find_map(|(i, family)| {
        let i = i as u32;
        let graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        // SAFETY: as above.
        let present = unsafe { surface_loader.get_physical_device_surface_support(pd, i, surface) }
            .unwrap_or(false);
        (graphics && present).then_some(i)
    })
}
