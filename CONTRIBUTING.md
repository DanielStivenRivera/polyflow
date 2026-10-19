# Contributing to Polyflow

Thanks for your interest! Polyflow is in early development, so the best way to
help right now is to read the [design document](docs/DESIGN.md), pick an open
issue, and say hi before starting on something large.

## Before opening a pull request

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CI runs the same checks on Windows and Linux.

## Guidelines

- **Performance is a feature.** Changes touching the reference-mesh path
  (import, rendering, BVH, baking) should include benchmark numbers on a large
  mesh. See the budgets in [docs/DESIGN.md](docs/DESIGN.md#6-performance-budgets).
- **Respect the crate boundaries** described in [ARCHITECTURE.md](ARCHITECTURE.md).
- **`unsafe` only in `polyflow-render`**, each block with a `// SAFETY:` comment.
- **Never panic on user data.** Bad files and bad input return errors.
- Keep pull requests focused; one change per PR.
- Write code, comments and commit messages in English.

## Shaders

Shaders are written in [Slang](https://shader-slang.org/) under
`crates/render/shaders/` and compiled to SPIR-V by the render crate's build
script, using `slangc` from the [Vulkan SDK](https://vulkan.lunarg.com/) (or
`$SLANGC`). Precompiled SPIR-V is committed in `crates/render/shaders/spirv/`
so the project builds without the SDK; when you change a shader with `slangc`
available, the build refreshes that folder — commit it with your change.

## Debugging the renderer

Vulkan validation layers are enabled in debug builds when the Vulkan SDK is
installed. Set `POLYFLOW_VALIDATION=0` or `1` to override, and
`POLYFLOW_GPU=<name substring>` to pick a specific GPU.

## Benchmarking

The app has a reproducible benchmark mode: it spins the camera around the mesh
with vsync off, then prints CPU, GPU and frame-interval statistics and exits.

```sh
cargo run --release -- --torus 7072 --bench 5            # 50M-triangle test torus
cargo run --release -- sculpt.obj --bench 5 --zoom 4     # your mesh, zoomed in 4x
```

Include before/after numbers from it in pull requests that touch rendering.
`POLYFLOW_GPU=<name>` selects the GPU, which is handy for testing integrated GPUs.
`POLYFLOW_AUTOSAVE_SECS=<n>` shortens the autosave interval (2 minutes by
default) to test crash recovery.

## Profiling

Use the `profiling` profile for optimized builds with debug info:

```sh
cargo build --profile profiling
```

## License

By contributing, you agree that your contributions will be dual licensed under
MIT and Apache-2.0, as described in the [README](README.md#license).
