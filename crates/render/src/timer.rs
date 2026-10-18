//! GPU frame timing with timestamp queries.

use std::time::Duration;

use ash::vk;

use crate::RenderError;
use crate::context::GpuContext;
use crate::renderer::FRAMES_IN_FLIGHT;

/// Measures the GPU time of each frame's command buffer. Disabled (all
/// methods no-ops) when the queue does not support timestamps.
#[derive(Debug)]
pub struct GpuTimer {
    pool: vk::QueryPool,
    nanos_per_tick: f64,
    /// Whether each frame slot has written timestamps that can be read back.
    written: [bool; FRAMES_IN_FLIGHT],
}

impl GpuTimer {
    pub fn new(ctx: &GpuContext) -> Result<Self, RenderError> {
        if ctx.timestamp_valid_bits == 0 {
            log::warn!("GPU timestamps not supported on this queue; GPU frame times unavailable");
            return Ok(Self {
                pool: vk::QueryPool::null(),
                nanos_per_tick: 0.0,
                written: [false; FRAMES_IN_FLIGHT],
            });
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(2 * FRAMES_IN_FLIGHT as u32);
        // SAFETY: valid device and create info.
        let pool = unsafe { ctx.device.create_query_pool(&info, None)? };
        Ok(Self {
            pool,
            nanos_per_tick: f64::from(ctx.properties.limits.timestamp_period),
            written: [false; FRAMES_IN_FLIGHT],
        })
    }

    fn enabled(&self) -> bool {
        self.pool != vk::QueryPool::null()
    }

    /// Reads the previous measurement of `frame`. Call after its fence has
    /// been waited on, before recording it again.
    pub fn collect(&mut self, device: &ash::Device, frame: usize) -> Option<Duration> {
        if !self.enabled() || !std::mem::take(&mut self.written[frame]) {
            return None;
        }
        let mut ticks = [0u64; 2];
        // SAFETY: both queries were written by a submission whose fence has signaled.
        let result = unsafe {
            device.get_query_pool_results(
                self.pool,
                2 * frame as u32,
                &mut ticks,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        result.ok()?;
        let nanos = ticks[1].wrapping_sub(ticks[0]) as f64 * self.nanos_per_tick;
        Some(Duration::from_nanos(nanos as u64))
    }

    pub fn begin(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, frame: usize) {
        if !self.enabled() {
            return;
        }
        // SAFETY: `cmd` is recording outside a rendering scope; the queries belong to `frame`.
        unsafe {
            device.cmd_reset_query_pool(cmd, self.pool, 2 * frame as u32, 2);
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::NONE,
                self.pool,
                2 * frame as u32,
            );
        }
    }

    pub fn end(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, frame: usize) {
        if !self.enabled() {
            return;
        }
        // SAFETY: as in `begin`.
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                self.pool,
                2 * frame as u32 + 1,
            );
        }
        self.written[frame] = true;
    }

    pub fn destroy(&mut self, ctx: &GpuContext) {
        if self.enabled() {
            // SAFETY: the caller guarantees the GPU is idle.
            unsafe { ctx.device.destroy_query_pool(self.pool, None) };
            self.pool = vk::QueryPool::null();
        }
    }
}
