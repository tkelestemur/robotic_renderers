import io
import subprocess

import pytest

from robotic_renderers.remote import (
    UV_DISCOVERY_SCRIPT,
    parse_linux_process_starttime,
    read_exact,
    rsync_command,
    rsync_remote,
    ssh_command,
    validate_ssh_host,
    worker_cleanup_script,
)


@pytest.mark.parametrize("location", ("path", "home", "missing"))
def test_uv_discovery_works_without_interactive_shell_path(tmp_path, location):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    for directory in (binary_dir, local_bin):
        if location == "missing" or (location == "home" and directory == binary_dir):
            continue
        executable = directory / "uv"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
    result = subprocess.run(
        ["/bin/sh", "-eu", "-c", UV_DISCOVERY_SCRIPT + '\nprintf "%s" "$uv_bin"'],
        env={"HOME": str(tmp_path), "PATH": str(binary_dir)},
        text=True,
        capture_output=True,
    )
    if location == "missing":
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stderr
        expected = binary_dir if location == "path" else local_bin
        assert result.stdout == str(expected / "uv")


@pytest.mark.parametrize(
    "host",
    ("render-host", "backup-render-host", "user@render.example.com", "gpu_01"),
)
def test_validate_ssh_host_accepts_aliases_and_hostnames(host):
    assert validate_ssh_host(host) == host


@pytest.mark.parametrize(
    "host",
    ("", "-oProxyCommand=bad", "host name", "host;touch /tmp/bad", "user@"),
)
def test_validate_ssh_host_rejects_option_and_shell_injection(host):
    with pytest.raises(ValueError):
        validate_ssh_host(host)


def test_ssh_command_has_noninteractive_timeouts_and_keepalives():
    command = ssh_command("render-host", "true")

    assert command[0] == "ssh"
    assert "BatchMode=yes" in command
    assert "ConnectTimeout=10" in command
    assert "ServerAliveInterval=15" in command
    assert command[-2:] == ["render-host", "true"]
    assert rsync_remote("render-host", "~/job") == "render-host:~/job"


def test_read_exact_rejects_oversized_and_truncated_payloads():
    with pytest.raises(RuntimeError, match="outside"):
        read_exact(io.BytesIO(b""), 11, max_bytes=10)
    with pytest.raises(EOFError, match="3 of 4"):
        read_exact(io.BytesIO(b"abc"), 4)
    assert read_exact(io.BytesIO(b"abcd"), 4) == b"abcd"


def test_rsync_command_uses_the_same_noninteractive_transport():
    command = rsync_command(
        "source/",
        rsync_remote("render-host", "~/job/"),
        delete=True,
        excludes=("__pycache__", "*.py[co]"),
    )

    assert command[:3] == ["rsync", "-az", "-e"]
    assert "BatchMode=yes" in command[3]
    assert "ConnectTimeout=10" in command[3]
    assert "ServerAliveInterval=15" in command[3]
    assert "--delete" in command
    assert "--exclude=__pycache__" in command
    assert command[-2:] == ["source/", "render-host:~/job/"]


def test_linux_stat_parser_handles_spaces_and_parentheses_in_process_name():
    fields_3_through_21 = ["S", *map(str, range(4, 22))]
    stat_line = f"123 (worker) with spaces) {' '.join(fields_3_through_21)} 987654 0 0"

    assert parse_linux_process_starttime(stat_line) == "987654"


def test_worker_cleanup_revalidates_identity_immediately_before_each_signal():
    script = worker_cleanup_script(
        "code/robot_rendering_mitsuba",
        ".jobs/example",
        "robotic_renderers.backends.mitsuba",
        "worker*.pid",
    )

    assert 'test -r "/proc/$candidate/stat"' in script
    assert 'actual_start=$(proc_start "$candidate")' in script
    assert '*"$module_marker"*"$job_marker"*' in script
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    for signal in ("TERM", "KILL"):
        group_prefix, fallback = script.split(
            f'kill -{signal} -- "-$candidate"',
            maxsplit=1,
        )
        signal_block = group_prefix.rsplit("for pid_file", maxsplit=1)[-1]
        assert signal_block.count('worker_matches "$pid_file"') == 2
        fallback_block = fallback.split(
            f'kill -{signal} "$candidate"',
            maxsplit=1,
        )[0]
        assert fallback_block.count('worker_matches "$pid_file"') == 2


@pytest.mark.parametrize(
    ("remote_job", "pid_pattern"),
    (("../escape", "worker.pid"), (".jobs/example", "../worker.pid")),
)
def test_worker_cleanup_rejects_unsafe_paths(remote_job, pid_pattern):
    with pytest.raises(ValueError, match="Unsafe"):
        worker_cleanup_script(
            "code/robot_rendering_mitsuba",
            remote_job,
            "robotic_renderers.backends.mitsuba",
            pid_pattern,
        )
