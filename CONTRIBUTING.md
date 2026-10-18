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

## Profiling

Use the `profiling` profile for optimized builds with debug info:

```sh
cargo build --profile profiling
```

## License

By contributing, you agree that your contributions will be dual licensed under
MIT and Apache-2.0, as described in the [README](README.md#license).
