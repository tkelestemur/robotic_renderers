"""Device-side RGB processing; only finished uint8 pixels cross back to the host."""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})
_Values8 = wp.types.vector(length=8, dtype=wp.float32)


@wp.func
def _byte(value: float):
    return wp.uint8(wp.rint(wp.clamp(value, 0.0, 255.0)))


@wp.kernel
def packed_rgb(
    packed: wp.array2d[wp.uint32],
    offsets: wp.array[int],
    camera: int,
    render_width: int,
    supersample: int,
    exposure: float,
    output: wp.array4d[wp.uint8],
):
    world, y, x = wp.tid()
    color = wp.vec3(0.0)
    for sy in range(supersample):
        for sx in range(supersample):
            pixel = offsets[camera] + (y * supersample + sy) * render_width + x * supersample + sx
            value = packed[world, pixel]
            # Match get_rgb's channel order and exposure/quantization before filtering.
            for channel in range(3):
                shift = wp.uint32((2 - channel) * 8)
                sample = float((value >> shift) & wp.uint32(255)) * (1.0 / 255.0)
                color[channel] += float(_byte(wp.clamp(sample * exposure, 0.0, 1.0) * 255.0))
    for channel in range(3):
        output[world, y, x, channel] = _byte(color[channel] / float(supersample * supersample))


@wp.kernel
def gamma_rgb(source: wp.array3d[wp.uint8], gamma: float, output: wp.array3d[wp.uint8]):
    y, x, channel = wp.tid()
    output[y, x, channel] = _byte(wp.pow(float(source[y, x, channel]) / 255.0, gamma) * 255.0)


@wp.kernel
def copy_rgb(source: wp.array3d[wp.uint8], output: wp.array3d[wp.uint8]):
    y, x, channel = wp.tid()
    output[y, x, channel] = source[y, x, channel]


@wp.func
def _neighbor_median(source: wp.array3d[float], y: int, x: int, channel: int):
    values = _Values8(0.0)
    index = int(0)  # noqa: UP018 - mutable loop variable in Warp's code generator
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy != 0 or dx != 0:
                sy = wp.clamp(y + dy, 0, source.shape[0] - 1)
                sx = wp.clamp(x + dx, 0, source.shape[1] - 1)
                values[index] = source[sy, sx, channel]
                index += 1
    for i in range(1, 8):
        for j in range(i, 0, -1):
            if values[j] < values[j - 1]:
                temporary = values[j]
                values[j] = values[j - 1]
                values[j - 1] = temporary
    return 0.5 * (values[3] + values[4])


@wp.kernel
def filmic_rgb(source: wp.array3d[float], exposure: float, output: wp.array3d[wp.uint8]):
    y, x = wp.tid()
    linear = wp.vec3(source[y, x, 0], source[y, x, 1], source[y, x, 2])
    median = wp.vec3(
        _neighbor_median(source, y, x, 0),
        _neighbor_median(source, y, x, 1),
        _neighbor_median(source, y, x, 2),
    )
    weights = wp.vec3(0.2126, 0.7152, 0.0722)
    if wp.dot(linear, weights) > 5.0 * wp.dot(median, weights) + 1.0 and wp.max(linear) > 8.0:
        linear = median
    for channel in range(3):
        value = wp.max(0.0, linear[channel] * exposure)
        mapped = (value * (2.51 * value + 0.03)) / (value * (2.43 * value + 0.59) + 0.14)
        mapped = wp.clamp(mapped, 0.0, 1.0)
        srgb = 12.92 * mapped
        if mapped > 0.0031308:
            srgb = 1.055 * wp.pow(mapped, 1.0 / 2.4) - 0.055
        output[y, x, channel] = _byte(srgb * 255.0)


@wp.kernel
def compose_tile(
    source: wp.array3d[wp.uint8],
    overlay: wp.array3d[wp.uint8],
    left: int,
    top: int,
    output: wp.array3d[wp.uint8],
):
    y, x, channel = wp.tid()
    alpha = float(overlay[top + y, left + x, 3]) / 255.0
    value = float(source[y, x, channel]) * (1.0 - alpha)
    value += float(overlay[top + y, left + x, channel]) * alpha
    output[top + y, left + x, channel] = _byte(value)


def pixel_device():
    """Use CUDA when available; the same kernels support portable CPU-only checkouts."""
    wp.init()
    return wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")


def rgb_buffer(width: int, height: int, *, device=None):
    return wp.empty((height, width, 3), dtype=wp.uint8, device=device or pixel_device())


def rgb_host_buffer(device_buffer):
    return wp.empty(
        device_buffer.shape,
        dtype=wp.uint8,
        device="cpu",
        pinned=device_buffer.device.is_cuda,
    )


def read_rgb(device_buffer, host_buffer):
    """Copy into reusable pinned storage. The returned view lasts until the next frame."""
    wp.copy(host_buffer, device_buffer)
    if device_buffer.device.is_cuda:
        wp.synchronize_device(device_buffer.device)
    return host_buffer.numpy()
