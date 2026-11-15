"""MuJoCo Warp shading with NVIDIA OptiX traversal in an isolated remote runtime."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from functools import partial

from robotic_renderers.backends import mujoco_warp as warp
from robotic_renderers.defaults import DEFAULT_RENDER_HOST
from robotic_renderers.remote import validate_ssh_host

BACKEND_NAME = "warp-optix"
DEFAULT_HOST = os.environ.get("ROBOT_RENDERING_WARP_OPTIX_HOST", DEFAULT_RENDER_HOST)
MUJOCO_WARP_COMMIT = "cafb631fbda1ff69ef9db0bd6c6bc57896dc889d"
MUJOCO_WARP_SOURCE = (
    f"https://github.com/google-deepmind/mujoco_warp/archive/{MUJOCO_WARP_COMMIT}.tar.gz"
)

# Share scene translation, shading, streaming, and transport with ordinary Warp.
# These defaults also allow direct local calls inside the isolated CUDA runtime.
render_frame = partial(warp.render_frame, host=DEFAULT_HOST, local=False, ray_backend="optix")
render_still = partial(warp.render_still, host=DEFAULT_HOST, local=False, ray_backend="optix")
render_live = partial(warp.render_live, host=DEFAULT_HOST, local=False, ray_backend="optix")
record_video = partial(warp.record_video, host=DEFAULT_HOST, local=False, ray_backend="optix")
benchmark = partial(warp.benchmark, host=DEFAULT_HOST, local=False, ray_backend="optix")


def setup_remote(host: str = DEFAULT_HOST, force: bool = False) -> dict[str, object]:
    return warp.setup_remote(host, force=force, ray_backend="optix")


def worker_status() -> dict[str, object]:
    """Probe the pinned build and actual OptiX context on the GPU worker."""
    distribution = importlib.metadata.distribution("mujoco-warp")
    provenance = json.loads(distribution.read_text("direct_url.json") or "{}")
    if provenance.get("url") != MUJOCO_WARP_SOURCE:
        raise RuntimeError("Unexpected MuJoCo Warp revision; rerun setup --backend warp-optix")
    context = warp.create_renderer(16, 16, supersample=1, ray_backend="optix")
    return {
        "backend": BACKEND_NAME,
        "available": True,
        "version": importlib.metadata.version("warp-optix"),
        "mujoco_warp_commit": MUJOCO_WARP_COMMIT,
        "ray_backend": context["render_context"].backend,
        "device": str(warp.wp.get_device(context["device"])),
        "execution": "remote",
        "live": True,
        "record": True,
    }


def backend_status(host: str = DEFAULT_HOST) -> dict[str, object]:
    host = validate_ssh_host(host)
    try:
        result = warp._run_remote_worker(host, ["status"], ray_backend="optix", timeout=60.0)
        return {**result, "host": host}
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        return {
            "backend": BACKEND_NAME,
            "available": False,
            "host": host,
            "execution": "remote",
            "notes": (
                "Requires the isolated warp-optix runtime and an NVIDIA CUDA/OptiX GPU. "
                "Run setup --backend warp-optix --host render-host."
            ),
            "error": str(detail or exc)[-3000:],
        }
