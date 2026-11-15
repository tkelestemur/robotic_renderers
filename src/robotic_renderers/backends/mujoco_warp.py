"""MuJoCo Warp ray-renderer backend for macOS CPU, with optional SSH CUDA."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from robotic_renderers.defaults import (
    DEFAULT_HEIGHT,
    DEFAULT_RECORDING_DURATION,
    DEFAULT_RENDER_HOST,
    DEFAULT_VIDEO_FPS,
    DEFAULT_WIDTH,
)
from robotic_renderers.media import (
    atomic_output_path,
    display_frames,
    image_digest,
    write_png,
    write_video,
)
from robotic_renderers.pixels import packed_rgb, read_rgb, rgb_host_buffer
from robotic_renderers.processes import terminate_and_reap
from robotic_renderers.remote import (
    UV_DISCOVERY_SCRIPT,
    read_exact,
    validate_ssh_host,
)
from robotic_renderers.scene import create_model_and_data, set_scene_time, trajectory_times

BACKEND_NAME = "warp"
WARP_EXPOSURE_SCALE = 0.42
DEFAULT_HOST = os.environ.get("ROBOT_RENDERING_WARP_HOST", DEFAULT_RENDER_HOST)
REMOTE_ROOT = ".cache/robot_rendering"
STREAM_MAGIC = b"ROBOT_RENDERING_RGB_STREAM_V1\n"
RESULT_MAGIC = "ROBOT_RENDERING_RESULT="
SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
)
REMOTE_BUNDLE_FILES = (
    ".python-version",
    "LICENSE",
    "THIRD_PARTY.md",
)
REMOTE_BUNDLE_EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "checkpoints",
    "dist",
    "models",
    "outputs",
    "videos",
    "weights",
}
REMOTE_BUNDLE_SENSITIVE_NAMES = {
    ".env",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
REMOTE_BUNDLE_SENSITIVE_SUFFIXES = (".key", ".p12", ".pem", ".pfx", ".pyc", ".pyo")
_SYNCED_HOSTS: dict[tuple[str, str], str] = {}
PACKAGE_SOURCE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_SOURCE.parents[1]


def _backend_name(ray_backend: str) -> str:
    if ray_backend not in ("warp", "optix"):
        raise ValueError(f"Unknown Warp ray backend: {ray_backend}")
    return "warp-optix" if ray_backend == "optix" else BACKEND_NAME


def _runtime_directory(ray_backend: str) -> Path:
    return Path("runtime") / _backend_name(ray_backend)


def _remote_root(ray_backend: str) -> str:
    if _backend_name(ray_backend) == "warp-optix":
        return ".cache/robot_rendering_warp_optix"
    return REMOTE_ROOT


def _run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def _ssh_command(host: str, *arguments: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, validate_ssh_host(host), *arguments]


def _rsync_ssh_transport() -> str:
    return shlex.join(["ssh", *SSH_OPTIONS])


def _bundle_path_is_safe(relative_path: Path) -> bool:
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return False
    if any(part in REMOTE_BUNDLE_EXCLUDED_PARTS for part in relative_path.parts):
        return False
    if any(part == ".env" or part.startswith(".env.") for part in relative_path.parts):
        return False
    name = relative_path.name
    if name in REMOTE_BUNDLE_SENSITIVE_NAMES:
        return False
    return not name.lower().endswith(REMOTE_BUNDLE_SENSITIVE_SUFFIXES)


def _remote_bundle_sources(root: Path | None = None, ray_backend: str = "warp") -> dict[Path, Path]:
    root = (root or PROJECT_ROOT).resolve()
    runtime_directory = _runtime_directory(ray_backend)
    required = (
        runtime_directory / "pyproject.toml",
        runtime_directory / "uv.lock",
        Path("src/robotic_renderers/backends/mujoco_warp.py"),
        Path("src/robotic_renderers/scene.py"),
    )
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Remote Warp bundle is missing required files: {missing}")

    sources: dict[Path, Path] = {}
    for name in REMOTE_BUNDLE_FILES:
        relative_path = Path(name)
        if (root / relative_path).is_file() and _bundle_path_is_safe(relative_path):
            sources[relative_path] = root / relative_path
    for relative_directory in (runtime_directory, Path("src"), Path("third_party")):
        directory = root / relative_directory
        if not directory.is_dir():
            continue
        for source in directory.rglob("*"):
            if source.is_file() or source.is_symlink():
                relative_path = source.relative_to(root)
                if _bundle_path_is_safe(relative_path):
                    sources[relative_path] = source
    return sources


def _remote_bundle_fingerprint(root: Path | None = None, ray_backend: str = "warp") -> str:
    digest = hashlib.sha256()
    sources = _remote_bundle_sources(root, ray_backend)
    for relative_path in sorted(sources, key=lambda path: path.as_posix()):
        source = sources[relative_path]
        digest.update(relative_path.as_posix().encode())
        if source.is_symlink():
            digest.update(os.readlink(source).encode())
            continue
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _stage_remote_bundle(
    destination: Path,
    root: Path | None = None,
    ray_backend: str = "warp",
) -> tuple[Path, ...]:
    """Copy only the manifest into an isolated rsync source directory."""
    destination.mkdir(parents=True, exist_ok=True)
    sources = _remote_bundle_sources(root, ray_backend)
    entries = tuple(sorted(sources, key=lambda path: path.as_posix()))
    for relative_path in entries:
        source = sources[relative_path]
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            link_target = Path(os.readlink(source))
            if link_target.is_absolute() or ".." in link_target.parts:
                raise ValueError(f"Unsafe symlink in remote Warp bundle: {relative_path}")
            target.symlink_to(link_target)
        else:
            shutil.copy2(source, target)
    return entries


def _remote_shell_command(arguments: list[str], ray_backend: str = "warp") -> str:
    remote_root = f"$HOME/{_remote_root(ray_backend)}"
    worker = shlex.join(
        [
            f"{remote_root}/.venv/bin/python",
            "-m",
            "robotic_renderers.backends.mujoco_warp",
            "__worker",
            "--ray-backend",
            ray_backend,
            *arguments,
        ]
    )
    for token in (f"{remote_root}/.venv/bin/python",):
        worker = worker.replace(shlex.quote(token), f'"{token}"')
    return (
        f'root="{remote_root}"; project="$root/source"; cd "$root" && '
        f'PYTHONPATH="$project/src" {worker}'
    )


def _remote_sync_script(incoming: str, ray_backend: str = "warp") -> str:
    return f"""set -euo pipefail
umask 077
root="$HOME/{_remote_root(ray_backend)}"
incoming="$root/{incoming}"
runtime="$incoming/{_runtime_directory(ray_backend).as_posix()}"
test -f "$runtime/pyproject.toml"
test -f "$runtime/uv.lock"
exec 9>"$root/sync.lock"
chmod 600 "$root/sync.lock"
flock -w 120 9
{UV_DISCOVERY_SCRIPT}
UV_PROJECT_ENVIRONMENT="$root/.venv" "$uv_bin" sync \
  --directory "$runtime" --python 3.12 --frozen --no-dev
PYTHONPATH="$incoming/src" "$root/.venv/bin/python" -c \
  'import imageio, mujoco, mujoco_warp, numpy, PIL, warp; from pxr import Sdf'
old="$root/source.old-$$"
if [ -e "$root/source" ]; then mv "$root/source" "$old"; fi
mv "$incoming" "$root/source"
rm -rf "$old"
"""


def backend_status(host: str = DEFAULT_HOST) -> dict[str, object]:
    # Keep Warp's initialization banner out of the CLI's JSON stdout.
    with contextlib.redirect_stdout(sys.stderr):
        local_device = (
            f"{wp.get_device('cuda:0').name} (cuda:0)"
            if wp.is_cuda_available()
            else f"CPU ({os.uname().machine})"
        )
    return {
        "backend": BACKEND_NAME,
        "available": True,
        "version": mjw.__version__,
        "device": local_device,
        "execution": "local",
        "live": True,
        "record": True,
        "notes": (
            "Warp uses local CUDA when available, otherwise CPU (including macOS). "
            f"Remote CUDA on {host} remains optional via --remote-warp."
        ),
    }


def setup_remote(
    host: str = DEFAULT_HOST, force: bool = False, *, ray_backend: str = "warp"
) -> dict[str, object]:
    host = validate_ssh_host(host)
    remote_root = _remote_root(ray_backend)
    fingerprint = _remote_bundle_fingerprint(ray_backend=ray_backend)
    cache_key = (host, ray_backend)
    if _SYNCED_HOSTS.get(cache_key) == fingerprint and not force:
        return {"host": host, "synced": True, "cached": True}

    uv_probe = subprocess.run(
        _ssh_command(host, UV_DISCOVERY_SCRIPT),
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    if uv_probe.returncode != 0:
        raise RuntimeError(
            f"uv is not installed on {host}. Install it with: "
            f"ssh {host} 'curl -LsSf https://astral.sh/uv/install.sh | sh'"
        )

    upload_id = uuid.uuid4().hex
    incoming = f"source.upload-{upload_id}"
    remote_incoming = f"{host}:~/{remote_root}/{incoming}/"
    _run(
        _ssh_command(
            host,
            f'umask 077; install -d -m 700 "$HOME/{remote_root}" '
            f'"$HOME/{remote_root}/.jobs" "$HOME/{remote_root}/{incoming}"; '
            f'chmod 700 "$HOME/{remote_root}" "$HOME/{remote_root}/.jobs" '
            f'"$HOME/{remote_root}/{incoming}"',
        )
    )
    try:
        with tempfile.TemporaryDirectory(prefix="robot-rendering-warp-bundle-") as temporary:
            bundle = Path(temporary) / "source"
            _stage_remote_bundle(bundle, ray_backend=ray_backend)
            _run(
                [
                    "rsync",
                    "-az",
                    "-e",
                    _rsync_ssh_transport(),
                    f"{bundle}/",
                    remote_incoming,
                ]
            )
        _run(
            _ssh_command(host, "bash -s"),
            input=_remote_sync_script(incoming, ray_backend),
            text=True,
        )
    finally:
        subprocess.run(
            _ssh_command(host, f'rm -rf "$HOME/{remote_root}/{incoming}"'),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if ray_backend == "optix":
        _run_remote_worker(host, ["status"], ray_backend=ray_backend)
    _SYNCED_HOSTS[cache_key] = fingerprint
    return {"host": host, "synced": True, "cached": False}


def _ray_backend_options(ray_backend: str) -> dict[str, str]:
    """Keep released Warp compatible while requiring explicit OptiX support."""
    _backend_name(ray_backend)
    if ray_backend == "warp":
        return {}
    if not wp.is_cuda_available() or not wp.get_device().is_cuda:
        raise RuntimeError("warp-optix requires an NVIDIA CUDA/OptiX GPU; CPU fallback is disabled")
    if "backend" not in inspect.signature(mjw.create_render_context).parameters:
        raise RuntimeError(
            "This MuJoCo Warp build lacks OptiX support. "
            "Run setup --backend warp-optix to install the pinned isolated runtime."
        )
    try:
        importlib.import_module("warp_optix")
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"warp-optix could not initialize: {exc}. The pinned OptiX 9.1 runtime "
            "requires NVIDIA driver R590 or newer. Check the driver and rerun "
            "setup --backend warp-optix."
        ) from exc
    return {"backend": "optix"}


def _check_ray_backend(render_context, ray_backend: str) -> None:
    if ray_backend == "optix" and getattr(render_context, "backend", None) != "optix":
        raise RuntimeError(
            "MuJoCo Warp fell back from OptiX to Warp. Check warp-optix, the NVIDIA driver, "
            "and scene support; refusing to label fallback rendering as warp-optix."
        )


def create_renderer(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    supersample: int = 1,
    device: str | None = None,
    *,
    ray_backend: str = "warp",
) -> dict[str, object]:
    if supersample < 1:
        raise ValueError("supersample must be at least 1")
    wp.init()
    if device is None:
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.set_device(device)
    backend_options = _ray_backend_options(ray_backend)

    render_width = width * supersample
    render_height = height * supersample
    model, host_data = create_model_and_data(width=render_width, height=render_height)
    warp_model = mjw.put_model(model)
    floor_material_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "floor_material")
    if floor_material_id >= 0:
        texture_repeats = warp_model.mat_texrepeat.numpy()
        # Warp's procedural checker sampler produces two checks per texture period,
        # while MuJoCo OpenGL treats a period as one floor tile.
        texture_repeats[:, floor_material_id] *= 0.5
        warp_model.mat_texrepeat.assign(texture_repeats)
    warp_data = mjw.put_data(model, host_data, nworld=1)
    render_context = mjw.create_render_context(
        model,
        nworld=1,
        cam_res=(render_width, render_height),
        render_rgb=True,
        render_depth=False,
        render_seg=False,
        use_textures=True,
        use_shadows=True,
        use_ambient_lighting=True,
        enabled_geom_groups=[0, 1, 2],
        render_skybox=True,
        enable_backface_culling=True,
        enable_specular=True,
        enable_emission=True,
        enable_per_light_ambient=True,
        **backend_options,
    )
    _check_ray_backend(render_context, ray_backend)
    rgb = wp.empty((1, height, width, 3), dtype=wp.uint8)
    return {
        "model": model,
        "host_data": host_data,
        "warp_model": warp_model,
        "warp_data": warp_data,
        "render_context": render_context,
        "rgb": rgb,
        "rgb_host": rgb_host_buffer(rgb),
        "device": device,
        "width": width,
        "height": height,
        "render_width": render_width,
        "render_height": render_height,
        "supersample": supersample,
        "ray_backend": ray_backend,
    }


def _copy_kinematics_to_warp(context: dict[str, object]) -> None:
    model = context["model"]
    host_data = context["host_data"]
    warp_data = context["warp_data"]
    warp_data.geom_xpos.assign(np.asarray(host_data.geom_xpos, dtype=np.float32)[None])
    if hasattr(warp_data, "geom_xquat"):
        warp_data.geom_xquat.assign(np.asarray(host_data.geom_xquat, dtype=np.float32)[None])
    else:
        warp_data.geom_xmat.assign(
            np.asarray(host_data.geom_xmat, dtype=np.float32).reshape(1, model.ngeom, 3, 3)
        )
    warp_data.cam_xpos.assign(np.asarray(host_data.cam_xpos, dtype=np.float32)[None])
    warp_data.cam_xmat.assign(
        np.asarray(host_data.cam_xmat, dtype=np.float32).reshape(1, model.ncam, 3, 3)
    )


def render_context_frame(context: dict[str, object], time_s: float = 0.0) -> np.ndarray:
    set_scene_time(context["model"], context["host_data"], time_s)
    _copy_kinematics_to_warp(context)
    mjw.refit_bvh(context["warp_model"], context["warp_data"], context["render_context"])
    mjw.render(context["warp_model"], context["warp_data"], context["render_context"])
    rc = context["render_context"]
    wp.launch(
        packed_rgb,
        dim=(1, context["height"], context["width"]),
        inputs=[
            rc.rgb_data,
            rc.rgb_adr,
            0,
            context["render_width"],
            context["supersample"],
            WARP_EXPOSURE_SCALE,
        ],
        outputs=[context["rgb"]],
        device=context["device"],
    )
    return read_rgb(context["rgb"], context["rgb_host"])[0]


def render_frame_local(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
    supersample: int = 1,
    device: str | None = None,
    *,
    ray_backend: str = "warp",
) -> np.ndarray:
    context = create_renderer(
        width, height, supersample=supersample, device=device, ray_backend=ray_backend
    )
    return render_context_frame(context, time_s=time_s)


def render_frame(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
    supersample: int = 1,
    host: str = DEFAULT_HOST,
    local: bool = True,
    *,
    ray_backend: str = "warp",
) -> np.ndarray:
    if local:
        return render_frame_local(width, height, time_s, supersample, ray_backend=ray_backend)
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="mujoco-warp-frame-") as temporary:
        output = Path(temporary) / "frame.png"
        render_still(
            output, width, height, time_s, supersample, host, local=False, ray_backend=ray_backend
        )
        return np.asarray(Image.open(output).convert("RGB"))


def render_still_local(
    output_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
    supersample: int = 1,
    device: str | None = None,
    *,
    ray_backend: str = "warp",
) -> dict[str, object]:
    started = time.perf_counter()
    image = render_frame_local(width, height, time_s, supersample, device, ray_backend=ray_backend)
    output = write_png(image, output_path)
    return {
        "backend": _backend_name(ray_backend),
        "output": str(output),
        "seconds": time.perf_counter() - started,
        "sha256": image_digest(image),
        "width": width,
        "height": height,
        "device": device or ("cuda:0" if wp.is_cuda_available() else "cpu"),
        "supersample": supersample,
    }


def _finite_frames(
    context: dict[str, object],
    duration: float,
    fps: int,
    timing: dict[str, float] | None = None,
):
    for time_s in trajectory_times(duration, fps):
        started = time.perf_counter()
        frame = render_context_frame(context, time_s)
        if timing is not None:
            timing["render_seconds"] = timing.get("render_seconds", 0.0) + (
                time.perf_counter() - started
            )
        yield frame


def record_video_local(
    output_path: str | Path,
    duration: float = DEFAULT_RECORDING_DURATION,
    fps: int = DEFAULT_VIDEO_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    supersample: int = 1,
    device: str | None = None,
    *,
    ray_backend: str = "warp",
) -> dict[str, object]:
    context = create_renderer(
        width, height, supersample=supersample, device=device, ray_backend=ray_backend
    )
    timing: dict[str, float] = {}
    output, frame_count, total_seconds = write_video(
        _finite_frames(context, duration=duration, fps=fps, timing=timing),
        output_path,
        fps=fps,
    )
    render_seconds = timing.get("render_seconds", 0.0)
    encode_seconds = max(0.0, total_seconds - render_seconds)
    return {
        "backend": _backend_name(ray_backend),
        "output": str(output),
        "frames": frame_count,
        "seconds": total_seconds,
        "render_seconds": render_seconds,
        "encode_seconds": encode_seconds,
        "video_fps": fps,
        "render_fps": frame_count / max(render_seconds, 1e-9),
        "width": width,
        "height": height,
        "device": context["device"],
        "supersample": supersample,
    }


def benchmark_local(
    frame_count: int = 30,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup: int = 3,
    supersample: int = 1,
    device: str | None = None,
    *,
    ray_backend: str = "warp",
) -> dict[str, object]:
    init_started = time.perf_counter()
    context = create_renderer(
        width, height, supersample=supersample, device=device, ray_backend=ray_backend
    )
    init_seconds = time.perf_counter() - init_started
    for index in range(warmup):
        render_context_frame(context, index / fps)
    started = time.perf_counter()
    last = None
    for index in range(frame_count):
        last = render_context_frame(context, index / fps)
    render_seconds = time.perf_counter() - started
    return {
        "backend": _backend_name(ray_backend),
        "device": context["device"],
        "width": width,
        "height": height,
        "supersample": supersample,
        "frames": frame_count,
        "init_seconds": init_seconds,
        "render_seconds": render_seconds,
        "render_fps": frame_count / max(render_seconds, 1e-9),
        "last_frame_sha256": image_digest(last),
    }


def _parse_remote_result(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MAGIC):
            return json.loads(line[len(RESULT_MAGIC) :])
    raise RuntimeError(f"Remote worker returned no result marker. Output:\n{stdout[-4000:]}")


def _run_remote_worker(
    host: str, arguments: list[str], *, ray_backend: str = "warp", timeout: float | None = None
) -> dict[str, object]:
    try:
        completed = _run(
            _ssh_command(host, _remote_shell_command(arguments, ray_backend)),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        detail = (str(exc.stdout or "") + "\n" + str(exc.stderr or ""))[-12000:]
        raise RuntimeError(
            f"{_backend_name(ray_backend)} worker failed on {host}:\n{detail}"
        ) from exc
    return _parse_remote_result(completed.stdout)


def _copy_remote_output(
    host: str, remote_path: str, output_path: str | Path, *, ray_backend: str = "warp"
) -> Path:
    with atomic_output_path(output_path) as (temporary, output):
        _run(
            [
                "scp",
                *SSH_OPTIONS,
                f"{validate_ssh_host(host)}:~/{_remote_root(ray_backend)}/{remote_path}",
                str(temporary),
            ]
        )
    return output


def _cleanup_remote_job(host: str, job_id: str, *, ray_backend: str = "warp") -> None:
    subprocess.run(
        _ssh_command(host, f'rm -rf "$HOME/{_remote_root(ray_backend)}/.jobs/{job_id}"'),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def render_still(
    output_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    time_s: float = 1.0,
    supersample: int = 1,
    host: str = DEFAULT_HOST,
    local: bool = True,
    *,
    ray_backend: str = "warp",
) -> dict[str, object]:
    if local:
        return render_still_local(
            output_path, width, height, time_s, supersample, ray_backend=ray_backend
        )
    setup_remote(host, ray_backend=ray_backend)
    job_id = uuid.uuid4().hex
    remote_output = f".jobs/{job_id}/frame.png"
    started = time.perf_counter()
    try:
        result = _run_remote_worker(
            host,
            [
                "still",
                "--output",
                remote_output,
                "--width",
                str(width),
                "--height",
                str(height),
                "--time",
                str(time_s),
                "--supersample",
                str(supersample),
            ],
            ray_backend=ray_backend,
        )
        output = _copy_remote_output(host, remote_output, output_path, ray_backend=ray_backend)
    finally:
        _cleanup_remote_job(host, job_id, ray_backend=ray_backend)
    return {
        **result,
        "output": str(output),
        "host": host,
        "roundtrip_seconds": time.perf_counter() - started,
    }


def record_video(
    output_path: str | Path,
    duration: float = DEFAULT_RECORDING_DURATION,
    fps: int = DEFAULT_VIDEO_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    supersample: int = 1,
    host: str = DEFAULT_HOST,
    local: bool = True,
    *,
    ray_backend: str = "warp",
) -> dict[str, object]:
    if local:
        return record_video_local(
            output_path,
            duration,
            fps,
            width,
            height,
            supersample=supersample,
            ray_backend=ray_backend,
        )
    setup_remote(host, ray_backend=ray_backend)
    job_id = uuid.uuid4().hex
    remote_output = f".jobs/{job_id}/video.mp4"
    started = time.perf_counter()
    try:
        result = _run_remote_worker(
            host,
            [
                "record",
                "--output",
                remote_output,
                "--duration",
                str(duration),
                "--fps",
                str(fps),
                "--width",
                str(width),
                "--height",
                str(height),
                "--supersample",
                str(supersample),
            ],
            ray_backend=ray_backend,
        )
        output = _copy_remote_output(host, remote_output, output_path, ray_backend=ray_backend)
    finally:
        _cleanup_remote_job(host, job_id, ray_backend=ray_backend)
    return {
        **result,
        "output": str(output),
        "host": host,
        "roundtrip_seconds": time.perf_counter() - started,
    }


def benchmark(
    frame_count: int = 30,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    warmup: int = 3,
    supersample: int = 1,
    host: str = DEFAULT_HOST,
    local: bool = True,
    *,
    ray_backend: str = "warp",
) -> dict[str, object]:
    if local:
        return benchmark_local(
            frame_count,
            width,
            height,
            fps,
            warmup,
            supersample=supersample,
            ray_backend=ray_backend,
        )
    setup_remote(host, ray_backend=ray_backend)
    result = _run_remote_worker(
        host,
        [
            "benchmark",
            "--frames",
            str(frame_count),
            "--width",
            str(width),
            "--height",
            str(height),
            "--fps",
            str(fps),
            "--warmup",
            str(warmup),
            "--supersample",
            str(supersample),
        ],
        ray_backend=ray_backend,
    )
    return {**result, "host": host}


def _remote_stream_frames(
    host: str,
    width: int,
    height: int,
    fps: int,
    duration: float | None,
    supersample: int,
    *,
    ray_backend: str = "warp",
):
    arguments = [
        "stream",
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--supersample",
        str(supersample),
    ]
    if duration is not None:
        arguments.extend(["--duration", str(duration)])
    process = subprocess.Popen(
        _ssh_command(host, _remote_shell_command(arguments, ray_backend)),
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
        start_new_session=True,
    )
    assert process.stdout is not None
    startup_lines: list[bytes] = []
    frame_count = 0
    expected_frames = None if duration is None else sum(1 for _ in trajectory_times(duration, fps))
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                return_code = process.poll()
                if return_code not in (None, 0):
                    raise RuntimeError(
                        f"Remote Warp stream worker exited with {return_code} before its handshake"
                    )
                raise RuntimeError(
                    "Remote stream ended before its handshake:\n"
                    + b"".join(startup_lines[-40:]).decode(errors="replace")
                )
            if line == STREAM_MAGIC:
                break
            startup_lines.append(line)
        expected = width * height * 3
        while True:
            length_bytes = process.stdout.read(4)
            if not length_bytes:
                break
            if len(length_bytes) != 4:
                length_bytes += read_exact(process.stdout, 4 - len(length_bytes), max_bytes=4)
            frame_length = struct.unpack("!I", length_bytes)[0]
            if frame_length != expected:
                raise RuntimeError(f"Remote stream sent {frame_length} bytes; expected {expected}")
            frame = np.frombuffer(
                read_exact(process.stdout, frame_length, max_bytes=expected),
                dtype=np.uint8,
            )
            frame_count += 1
            if expected_frames is not None and frame_count > expected_frames:
                raise RuntimeError(
                    f"Remote Warp stream produced more than {expected_frames} frames"
                )
            yield frame.reshape(height, width, 3).copy()
        try:
            return_code = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Remote Warp stream closed stdout but did not exit within 5 seconds"
            ) from exc
        if return_code != 0:
            raise RuntimeError(f"Remote Warp stream worker exited with {return_code}")
        if expected_frames is not None and frame_count != expected_frames:
            raise RuntimeError(
                f"Remote Warp stream produced {frame_count} frames; expected {expected_frames}"
            )
    finally:
        terminate_and_reap({"warp": process}, grace_seconds=3.0)


def render_live(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_VIDEO_FPS,
    duration: float | None = None,
    supersample: int = 1,
    host: str = DEFAULT_HOST,
    local: bool = True,
    *,
    ray_backend: str = "warp",
) -> dict[str, object]:
    if local:
        context = create_renderer(width, height, supersample=supersample, ray_backend=ray_backend)
        device = context["device"]

        def frames():
            frame_index = 0
            while duration is None or frame_index / fps < duration:
                yield render_context_frame(context, frame_index / fps)
                frame_index += 1

        result = display_frames(
            frames(), width, height, f"{_backend_name(ray_backend)} ({device})", target_fps=fps
        )
        return {"backend": _backend_name(ray_backend), "device": device, **result}

    setup_remote(host, ray_backend=ray_backend)
    result = display_frames(
        _remote_stream_frames(
            host, width, height, fps, duration, supersample, ray_backend=ray_backend
        ),
        width,
        height,
        f"{_backend_name(ray_backend)} (CUDA on {host})",
        target_fps=fps,
    )
    return {"backend": _backend_name(ray_backend), "host": host, **result}


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("mode", choices=["still", "record", "benchmark", "stream", "status"])
    parser.add_argument("--ray-backend", choices=["warp", "optix"], default="warp")
    parser.add_argument("--output")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--supersample", type=int, default=1)
    return parser


def _worker_stream(args: argparse.Namespace) -> None:
    context = create_renderer(
        args.width,
        args.height,
        args.supersample,
        device="cuda:0",
        ray_backend=args.ray_backend,
    )
    render_context_frame(context, 0.0)  # JIT and allocate before the binary handshake.
    # Warp writes startup/JIT logs through the text wrapper; drain them before raw bytes.
    sys.stdout.flush()
    sys.stdout.buffer.write(STREAM_MAGIC)
    sys.stdout.buffer.flush()

    def stream_times():
        if args.duration is not None:
            yield from trajectory_times(args.duration, args.fps)
            return
        frame_index = 0
        while True:
            yield frame_index / args.fps
            frame_index += 1

    try:
        for time_s in stream_times():
            frame = render_context_frame(context, time_s)
            payload = frame.tobytes()
            sys.stdout.buffer.write(struct.pack("!I", len(payload)))
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
    except BrokenPipeError:
        return


def _worker_main(arguments: list[str]) -> int:
    args = _worker_parser().parse_args(arguments)
    if args.mode == "status":
        from robotic_renderers.backends.mujoco_warp_optix import worker_status

        print(RESULT_MAGIC + json.dumps(worker_status(), sort_keys=True), flush=True)
        return 0
    if args.mode == "stream":
        _worker_stream(args)
        return 0
    if args.mode == "still":
        if not args.output:
            raise SystemExit("--output is required")
        result = render_still_local(
            args.output,
            args.width,
            args.height,
            args.time,
            args.supersample,
            device="cuda:0",
            ray_backend=args.ray_backend,
        )
    elif args.mode == "record":
        if not args.output or args.duration is None:
            raise SystemExit("--output and --duration are required")
        result = record_video_local(
            args.output,
            args.duration,
            args.fps,
            args.width,
            args.height,
            args.supersample,
            device="cuda:0",
            ray_backend=args.ray_backend,
        )
    else:
        result = benchmark_local(
            args.frames,
            args.width,
            args.height,
            args.fps,
            args.warmup,
            args.supersample,
            device="cuda:0",
            ray_backend=args.ray_backend,
        )
    print(RESULT_MAGIC + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "__worker":
        raise SystemExit(_worker_main(sys.argv[2:]))
    raise SystemExit("Use `robotic-renderers`; __worker is reserved for the SSH runner.")
