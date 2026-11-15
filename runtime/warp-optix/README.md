# MuJoCo Warp + OptiX runtime

This frozen Python 3.12, Linux x86-64 environment powers `--backend warp-optix`.
It installs `warp-optix==0.3.0` / `pyoptix==9.1.1` and MuJoCo Warp commit
`cafb631fbda1ff69ef9db0bd6c6bc57896dc889d` (the OptiX integration merged on
2026-11-14). The integration is not in the published MuJoCo Warp 4.0.1 wheel.

The immutable source archive is hash-locked. Using an archive avoids inheriting
upstream's development-only package indexes; shared packages remain on the same
published versions as `runtime/warp`. Setup and status verify installed source
provenance and create an actual OptiX context. Rendering rejects upstream fallback.

From the repository root:

```bash
uv run robotic-renderers setup --backend warp-optix --host render-host
uv run robotic-renderers status --backend warp-optix --host render-host
uv run robotic-renderers still --backend warp-optix --host render-host \
  --width 320 --height 180 --supersample 1 --output outputs/warp-optix.png
```

The remote root is `~/.cache/robot_rendering_warp_optix`, separate from ordinary
Warp's environment. NVIDIA driver R590 or newer supporting CUDA and OptiX 9.1 is required;
the CUDA Toolkit and OptiX SDK are not separately needed. Installation and rendering
run on the remote Linux host, not the macOS orchestrator.

Refresh the lock independently:

```bash
uv lock --project runtime/warp-optix
```

Local checks cover dependency resolution, orchestration, fallback rejection, and
the pinned source's ordinary CPU rendering path. OptiX stills, short recordings,
headless live streams, and 1-/2-world batches have also passed smoke checks on an
NVIDIA L4 with driver 595.99.02.
