"""CPU-only checks for OptiX selection, runtime isolation, and orchestration."""

import json
import subprocess
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from robotic_renderers import cli
from robotic_renderers.backends import mujoco_warp as warp
from robotic_renderers.backends import mujoco_warp_optix as optix
from robotic_renderers.benchmarks import throughput
from robotic_renderers.scene import create_model_and_data, set_scene_time
from robotic_renderers.workflows import cosmos3


def test_optix_runtime_pins_source_revision_and_wheel():
    root = Path(__file__).resolve().parents[1] / "runtime" / "warp-optix"
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    packages = {package["name"]: package for package in lock["package"]}
    assert project["tool"]["uv"]["sources"]["mujoco-warp"]["url"] == optix.MUJOCO_WARP_SOURCE
    assert packages["mujoco-warp"]["source"]["url"] == optix.MUJOCO_WARP_SOURCE
    assert packages["mujoco-warp"]["sdist"]["hash"].startswith("sha256:")
    assert packages["warp-optix"]["version"] == "0.3.0"
    assert packages["pyoptix"]["version"] == "9.1.1"
    assert project["tool"]["uv"]["environments"] == [
        "sys_platform == 'linux' and platform_machine == 'x86_64'"
    ]


def test_optix_rejects_cpu_before_translating_scene(monkeypatch):
    monkeypatch.setattr(warp.wp, "is_cuda_available", lambda: False)
    build_scene = Mock(side_effect=AssertionError("scene must not be built"))
    monkeypatch.setattr(warp, "create_model_and_data", build_scene)
    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        warp.create_renderer(16, 16, device="cpu", ray_backend="optix")


def test_optix_requires_new_upstream_api(monkeypatch):
    monkeypatch.setattr(warp.wp, "is_cuda_available", lambda: True)
    monkeypatch.setattr(warp.wp, "get_device", lambda: SimpleNamespace(is_cuda=True))
    monkeypatch.setattr(warp.mjw, "create_render_context", lambda model: None)
    with pytest.raises(RuntimeError, match="pinned isolated runtime"):
        warp._ray_backend_options("optix")
    monkeypatch.setattr(warp.mjw, "create_render_context", lambda model, backend="warp": None)
    monkeypatch.setattr(warp.importlib, "import_module", Mock())
    assert warp._ray_backend_options("optix") == {"backend": "optix"}


def test_optix_reports_driver_abi_failure_before_upstream_hides_it(monkeypatch):
    monkeypatch.setattr(warp.wp, "is_cuda_available", lambda: True)
    monkeypatch.setattr(warp.wp, "get_device", lambda: SimpleNamespace(is_cuda=True))
    monkeypatch.setattr(warp.mjw, "create_render_context", lambda model, backend="warp": None)
    monkeypatch.setattr(
        warp.importlib, "import_module", Mock(side_effect=ImportError("Unsupported ABI version"))
    )
    with pytest.raises(RuntimeError, match="Unsupported ABI version.*R590"):
        warp._ray_backend_options("optix")


def test_remote_worker_failure_keeps_driver_diagnostics(monkeypatch):
    monkeypatch.setattr(
        warp,
        "_run",
        Mock(side_effect=subprocess.CalledProcessError(1, "ssh", stderr="Unsupported ABI version")),
    )
    with pytest.raises(RuntimeError, match="Unsupported ABI version"):
        warp._run_remote_worker("render-host", ["status"], ray_backend="optix")


def test_optix_rejects_upstream_fallback():
    with pytest.raises(RuntimeError, match="fell back"):
        warp._check_ray_backend(SimpleNamespace(backend="warp"), "optix")
    warp._check_ray_backend(SimpleNamespace(backend="optix"), "optix")


@pytest.mark.parametrize("quaternions", (False, True))
def test_animated_kinematics_support_both_warp_versions(quaternions):
    model, data = create_model_and_data(16, 16)
    geometry_field = "geom_xquat" if quaternions else "geom_xmat"
    fields = ("geom_xpos", geometry_field, "cam_xpos", "cam_xmat")
    buffers = {name: SimpleNamespace(assign=Mock()) for name in fields}
    context = {"model": model, "host_data": data, "warp_data": SimpleNamespace(**buffers)}
    for time_s in (0.0, 3.0):
        set_scene_time(model, data, time_s)
        warp._copy_kinematics_to_warp(context)
        for name, buffer in buffers.items():
            actual = buffer.assign.call_args.args[0]
            expected = getattr(data, name).astype(np.float32)
            np.testing.assert_array_equal(actual.reshape(expected.shape), expected)


def test_optix_status_checks_provenance_and_actual_context(monkeypatch):
    distribution = SimpleNamespace(read_text=lambda name: json.dumps({"url": "wrong"}))
    monkeypatch.setattr(optix.importlib.metadata, "distribution", lambda name: distribution)
    with pytest.raises(RuntimeError, match="revision"):
        optix.worker_status()
    distribution.read_text = lambda name: json.dumps({"url": optix.MUJOCO_WARP_SOURCE})
    create = Mock(
        return_value={"render_context": SimpleNamespace(backend="optix"), "device": "cpu"}
    )
    monkeypatch.setattr(warp, "create_renderer", create)
    monkeypatch.setattr(optix.importlib.metadata, "version", lambda name: "0.3.0")
    status = optix.worker_status()
    assert status["ray_backend"] == "optix"
    assert status["mujoco_warp_commit"] == optix.MUJOCO_WARP_COMMIT
    assert create.call_args.kwargs["ray_backend"] == "optix"


def test_optix_status_reports_unavailable_without_installing(monkeypatch):
    probe = Mock(side_effect=subprocess.CalledProcessError(1, "ssh", stderr="No CUDA driver"))
    monkeypatch.setattr(warp, "_run_remote_worker", probe)
    monkeypatch.setattr(warp, "setup_remote", Mock(side_effect=AssertionError("no setup")))
    result = optix.backend_status("render-host")
    assert result["available"] is False
    assert result["backend"] == "warp-optix"
    assert result["error"] == "No CUDA driver"
    assert probe.call_args.kwargs["ray_backend"] == "optix"


@pytest.mark.parametrize(
    ("command", "function"),
    (
        ("still", "render_still"),
        ("live", "render_live"),
        ("record", "record_video"),
        ("benchmark", "benchmark"),
    ),
)
def test_cli_optix_rendering_is_remote_and_uses_supersampling(monkeypatch, command, function):
    render = Mock(return_value={"backend": "warp-optix"})
    monkeypatch.setattr(cli, "_module", lambda name: SimpleNamespace(**{function: render}))
    assert (
        cli.main(
            [command, "--backend", "warp-optix", "--supersample", "3", "--host", "render-host"]
        )
        == 0
    )
    assert render.call_args.kwargs["host"] == "render-host"
    assert render.call_args.kwargs["local"] is False
    assert render.call_args.kwargs["supersample"] == 3
    assert "samples" not in render.call_args.kwargs


@pytest.mark.parametrize("mode", ("still", "record", "benchmark"))
def test_remote_optix_jobs_use_isolated_transport(monkeypatch, tmp_path, mode):
    setup = Mock()
    run = Mock(return_value={"backend": "warp-optix"})
    copy = Mock(return_value=tmp_path / "result")
    cleanup = Mock()
    monkeypatch.setattr(warp, "setup_remote", setup)
    monkeypatch.setattr(warp, "_run_remote_worker", run)
    monkeypatch.setattr(warp, "_copy_remote_output", copy)
    monkeypatch.setattr(warp, "_cleanup_remote_job", cleanup)
    if mode == "benchmark":
        result = optix.benchmark(host="render-host")
    elif mode == "still":
        result = optix.render_still(tmp_path / "result.png", host="render-host")
    else:
        result = optix.record_video(tmp_path / "result.mp4", host="render-host")
    assert result["backend"] == "warp-optix"
    assert setup.call_args.kwargs["ray_backend"] == "optix"
    assert run.call_args.args[1][0] == mode
    assert run.call_args.kwargs["ray_backend"] == "optix"
    if mode != "benchmark":
        assert copy.call_args.kwargs["ray_backend"] == "optix"
        assert cleanup.call_args.kwargs["ray_backend"] == "optix"
    command = warp._remote_shell_command(run.call_args.args[1], "optix")
    assert "--ray-backend optix" in command
    assert "$HOME/.cache/robot_rendering_warp_optix/.venv/bin/python" in command


def test_setup_cache_is_separate_for_warp_and_optix(monkeypatch):
    monkeypatch.setattr(warp, "_SYNCED_HOSTS", {})
    monkeypatch.setattr(warp, "_remote_bundle_fingerprint", lambda **kwargs: "same")
    monkeypatch.setattr(warp, "_stage_remote_bundle", Mock())
    monkeypatch.setattr(warp, "_run", Mock())
    monkeypatch.setattr(warp.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    probe = Mock()
    monkeypatch.setattr(warp, "_run_remote_worker", probe)
    assert warp.setup_remote("render-host")["cached"] is False
    assert optix.setup_remote("render-host")["cached"] is False
    assert optix.setup_remote("render-host")["cached"] is True
    assert warp.setup_remote("render-host")["cached"] is True
    assert probe.call_count == 1
    script = warp._remote_sync_script("source.upload-test", "optix")
    assert 'root="$HOME/.cache/robot_rendering_warp_optix"' in script
    assert 'runtime="$incoming/runtime/warp-optix"' in script
    assert "--frozen --no-dev" in script


@pytest.mark.parametrize("video", (False, True))
def test_cosmos_optix_sources_are_remote(monkeypatch, tmp_path, video):
    render = Mock(return_value={})
    module = SimpleNamespace(render_still=render, record_video=render)
    monkeypatch.setattr(cosmos3, "load_backend", lambda name: module)
    if video:
        cosmos3._render_source_video(
            "warp-optix", tmp_path / "source.mp4", 16, 16, 3, 6, 4, "render-host"
        )
    else:
        cosmos3._render_source_still(
            "warp-optix", tmp_path / "source.png", 16, 16, 4, "render-host"
        )
    assert render.call_args.kwargs["local"] is False
    assert render.call_args.kwargs["host"] == "render-host"


def test_throughput_sweeps_use_separate_warp_runtimes(monkeypatch):
    args = throughput.build_parser().parse_args(
        ["--backends", "warp", "warp-optix", "--warp-batches", "1,4", "--no-setup"]
    )
    commands = []

    def remote(host, command, timeout=None):
        commands.append(command)
        if "__hardware" in command:
            return {"gpus": [{"index": 0}], "logical_cpus": 8}
        if "__prepare" in command:
            return {}
        return {"render_fps": 1.0, "throughput_fps": 1.0}

    monkeypatch.setattr(throughput, "_remote_result", remote)
    result = throughput.run_suite(args)
    assert set(result["results"]) == {"warp", "warp-optix"}
    optix_commands = [command for command in commands if "--ray-backend optix" in command]
    assert len(optix_commands) == 2
    assert all('cd "$HOME/.cache/robot_rendering_warp_optix"' in cmd for cmd in optix_commands)
    warp_commands = [command for command in commands if "--ray-backend warp" in command]
    assert len(warp_commands) == 2
    assert all('cd "$HOME/.cache/robot_rendering"' in cmd for cmd in warp_commands)
