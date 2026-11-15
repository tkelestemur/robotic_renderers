"""Check device kernels against the previous color response and known image layouts."""

import numpy as np
import pytest
import warp as wp

from robotic_renderers import pixels

DEVICES = ["cpu", *(["cuda:0"] if wp.is_cuda_available() else [])]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("supersample", [1, 2, 3])
def test_packed_rgb_preserves_channels_worlds_offsets_and_exposure(device, supersample):
    rng = np.random.default_rng(42)
    source = rng.integers(0, 256, (2, 3 * supersample, 5 * supersample, 3), dtype=np.uint8)
    values = source.astype(np.uint32)
    packed = (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]
    packed = np.pad(packed.reshape(2, -1), ((0, 0), (7, 0)), constant_values=0xFF00FF)
    output = wp.empty((2, 3, 5, 3), dtype=wp.uint8, device=device)
    wp.launch(
        pixels.packed_rgb,
        dim=(2, 3, 5),
        inputs=[
            wp.array(packed, dtype=wp.uint32, device=device),
            wp.array([0, 7], dtype=int, device=device),
            1,
            5 * supersample,
            supersample,
            0.42,
        ],
        outputs=[output],
        device=device,
    )
    reference = np.rint(source.astype(np.float32) * (1 / 255) * 0.42 * 255)
    reference = reference.reshape(2, 3, supersample, 5, supersample, 3).mean(axis=(2, 4))
    np.testing.assert_array_equal(output.numpy(), np.rint(reference).astype(np.uint8))
    assert output.dtype == wp.uint8
    assert output.size == 2 * 3 * 5 * 3


@pytest.mark.parametrize("device", DEVICES)
def test_gamma_matches_previous_response_for_every_input_byte(device):
    source = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    output = pixels.rgb_buffer(256, 1, device=device)
    wp.launch(
        pixels.gamma_rgb,
        dim=output.shape,
        inputs=[wp.array(source, dtype=wp.uint8, device=device), 0.85],
        outputs=[output],
        device=device,
    )
    reference = np.rint(np.power(source.astype(np.float32) / 255, 0.85) * 255).astype(np.uint8)
    np.testing.assert_array_equal(output.numpy(), reference)


@pytest.mark.parametrize("device", DEVICES)
def test_filmic_kernel_matches_firefly_filter_and_tone_curve(device):
    source = np.random.default_rng(7).uniform(0, 4, (7, 9, 3)).astype(np.float32)
    source[0, 0] = 10000  # isolated outlier at a clamped border
    source[3:6, 3:6] = 100  # supported highlight must survive
    padded = np.pad(source, ((1, 1), (1, 1), (0, 0)), mode="edge")
    neighbors = [
        padded[y : y + 7, x : x + 9] for y in range(3) for x in range(3) if (y, x) != (1, 1)
    ]
    median = np.median(neighbors, axis=0)
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    outliers = (source @ weights > 5 * (median @ weights) + 1) & (source.max(axis=2) > 8)
    linear = np.maximum(0, np.where(outliers[..., None], median, source) * 0.36)
    mapped = (linear * (2.51 * linear + 0.03)) / (linear * (2.43 * linear + 0.59) + 0.14)
    mapped = np.clip(mapped, 0, 1)
    srgb = np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * mapped ** (1 / 2.4) - 0.055)
    reference = np.rint(srgb * 255).astype(np.uint8)
    output = pixels.rgb_buffer(9, 7, device=device)
    wp.launch(
        pixels.filmic_rgb,
        dim=(7, 9),
        inputs=[wp.array(source, dtype=float, device=device), 0.36],
        outputs=[output],
        device=device,
    )
    np.testing.assert_allclose(output.numpy(), reference, atol=1, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_composition_places_tiles_and_blends_static_overlay(device):
    source = np.full((2, 3, 3), [20, 40, 60], dtype=np.uint8)
    overlay = np.zeros((4, 6, 4), dtype=np.uint8)
    overlay[2, 3] = [100, 120, 140, 255]
    overlay[3, 4] = [100, 120, 140, 128]
    output = wp.zeros((4, 6, 3), dtype=wp.uint8, device=device)
    wp.launch(
        pixels.compose_tile,
        dim=source.shape,
        inputs=[
            wp.array(source, dtype=wp.uint8, device=device),
            wp.array(overlay, dtype=wp.uint8, device=device),
            3,
            2,
        ],
        outputs=[output],
        device=device,
    )
    actual = pixels.read_rgb(output, pixels.rgb_host_buffer(output))
    np.testing.assert_array_equal(actual[:2], 0)
    np.testing.assert_array_equal(actual[2, 3], [100, 120, 140])
    np.testing.assert_array_equal(actual[3, 4], [60, 80, 100])
    np.testing.assert_array_equal(actual[3, 5], [20, 40, 60])
