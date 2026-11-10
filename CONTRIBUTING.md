# Releasing

One version covers all three packages: `[workspace.package] version` in the root `Cargo.toml`.
The Rust crate `lineshapes` (crates.io), the Python package `lineshapes` (PyPI, version read from the
same place via `dynamic = ["version"]`), and the WebAssembly/TypeScript package `lineshapes` (npm,
version synced automatically from `Cargo.toml` by `crates/lineshapes-wasm/scripts/build.ts`) always
release together.

## To release

1. Bump `version` in `[workspace.package]` (root `Cargo.toml`).
2. Run `cargo update --workspace` (refreshes `Cargo.lock`; CI builds with `--locked`) and `make wasm` (syncs `crates/lineshapes-wasm/package.json`), then commit.
3. Push / merge to `main`.

`.github/workflows/release.yml` then: checks that tag `vX.Y.Z` does not exist yet -> runs the full
test suite (`ci.yml`) -> builds abi3 wheels (linux x86_64/aarch64, macOS x86_64/arm64, windows x64),
an sdist, and the npm WASM tarball -> publishes the crate to crates.io, the wheels to PyPI, and the
WASM package to npm -> creates tag `vX.Y.Z` and a GitHub release with the artifacts attached. Pushes
that do not change the version do nothing (the check job finds the existing tag). Publishing is
idempotent: a re-run skips versions already on crates.io / PyPI / npm, and the tag is only created
after all three succeed, so a failed release is fixed by re-running the workflow. `workflow_dispatch`
with `dry_run` builds everything without publishing.

## One-time setup (repository owner)

- License: the code is under the Berkeley Lab modified BSD license (`LICENSE`, `COPYRIGHT`; SPDX
  `BSD-3-Clause-LBNL`), as for other ALS software. The license and repository metadata are set in the
  workspace `Cargo.toml`, `pyproject.toml`, and `crates/lineshapes-wasm/package.json`. Release is
  subject to Berkeley Lab's required approvals (see the notice in `COPYRIGHT`; questions go to
  IPO@lbl.gov).
- crates.io: create an API token scoped to publish `lineshapes`, add it as the repository
  (or `crates-io` environment) secret `CARGO_REGISTRY_TOKEN`. The first publish of a new crate
  needs a token with the `publish-new` scope.
- PyPI: add a trusted publisher (pending publisher for a new project) for this repository, workflow
  `release.yml`, environment `pypi`. No token is stored.
- npm: after the initial local publish of `lineshapes`, go to the package's Settings on npmjs.com and
  add a GitHub Actions trusted publisher for `ALS-RSOXS/lineshapes`, workflow `release.yml`,
  environment `npm`. No token is stored once OIDC trusted publishing is enabled.
- GitHub: create the environments `pypi`, `crates-io`, and `npm` (optionally with required reviewers
  for a manual approval gate before publishing). Allow Actions to create releases (default token has
  `contents: write` only in the release job).

## Notes

- Versions already published cannot be overwritten; to fix a bad release, bump the version.
- `speed` tests are not run in CI (timing on shared runners is noisy). Run them locally with
  `uv run pytest -m speed` before releasing.
