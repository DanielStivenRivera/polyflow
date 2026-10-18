//! Frame time statistics for `--bench`.

use std::time::{Duration, Instant};

/// Frames rendered before this are ignored (shader warm-up, first uploads).
const WARMUP: Duration = Duration::from_secs(1);

#[derive(Debug)]
pub struct Bench {
    start: Instant,
    duration: Duration,
    cpu_ms: Vec<f32>,
    gpu_ms: Vec<f32>,
    frame_ms: Vec<f32>,
}

impl Bench {
    pub fn new(duration: Duration) -> Self {
        Self {
            start: Instant::now(),
            duration,
            cpu_ms: vec![],
            gpu_ms: vec![],
            frame_ms: vec![],
        }
    }

    /// Records one frame. Returns true once the benchmark is over.
    pub fn record(&mut self, cpu_ms: f32, gpu_ms: Option<f32>, frame_ms: Option<f32>) -> bool {
        let elapsed = self.start.elapsed();
        if elapsed >= WARMUP {
            self.cpu_ms.push(cpu_ms);
            self.gpu_ms.extend(gpu_ms);
            self.frame_ms.extend(frame_ms);
        }
        elapsed >= WARMUP + self.duration
    }

    pub fn report(&mut self) -> String {
        let mut out = format!(
            "benchmark over {:.1} s, {} frames\n",
            self.duration.as_secs_f32(),
            self.cpu_ms.len()
        );
        for (name, samples) in [
            ("frame interval", &mut self.frame_ms),
            ("GPU frame", &mut self.gpu_ms),
            ("CPU frame", &mut self.cpu_ms),
        ] {
            if samples.is_empty() {
                out += &format!("  {name:<15} n/a\n");
                continue;
            }
            samples.sort_by(f32::total_cmp);
            let mean = samples.iter().sum::<f32>() / samples.len() as f32;
            let p95 = samples[(samples.len() - 1) * 95 / 100];
            out += &format!(
                "  {name:<15} mean {mean:7.3} ms   p95 {p95:7.3} ms   max {:7.3} ms\n",
                samples.last().unwrap()
            );
        }
        out
    }
}
