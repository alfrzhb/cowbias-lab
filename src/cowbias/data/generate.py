"""Procedural animal and landscape originals, independent of their eventual pairings."""

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from cowbias.data.splits import Source, keyed_seed

GENERATOR_VERSION = "1.0"


def render_foreground(source: Source, size: int) -> Image.Image:
    """Render a reusable RGBA animal; placement and appearance depend only on its ID.

    Both species share palette, scale, position, outline, sampling and orientation
    distributions. Anatomical features and coat patterns distinguish the species.
    """
    if source.kind != "foreground" or source.label not in ("cow", "non_cow"):
        raise ValueError(f"Unsupported foreground source: {source}")
    rng = np.random.default_rng(keyed_seed(source.seed, source.source_id))
    unit = 3
    canvas = Image.new("RGBA", (size * unit, size * unit))
    draw = ImageDraw.Draw(canvas)
    # Use a common 100-unit anatomical canvas to support arbitrary resolutions.
    factor = size * unit / 100
    scale = float(rng.uniform(0.78, 1.04))
    dx, dy = rng.uniform(-5, 4), rng.uniform(-3, 4)

    def points(values: list[tuple[float, float]]) -> list[tuple[int, int]]:
        return [
            (
                round(((x - 50) * scale + 50 + dx) * factor),
                round(((y - 56) * scale + 56 + dy) * factor),
            )
            for x, y in values
        ]

    def ellipse(box: tuple[float, float, float, float], color: tuple) -> None:
        draw.ellipse(points([(box[0], box[1]), (box[2], box[3])]), fill=color)

    def line(vertices: list[tuple[float, float]], color: tuple, width: float) -> None:
        draw.line(points(vertices), fill=color, width=max(1, round(width * factor * scale)))

    def polygon(vertices: list[tuple[float, float]], color: tuple) -> None:
        draw.polygon(points(vertices), fill=color)

    cow = source.label == "cow"
    palette = [(221, 205, 177), (168, 115, 72), (88, 66, 53), (229, 224, 206), (121, 91, 70)]
    coat = palette[int(rng.integers(len(palette)))]
    shade = tuple(max(15, c - 37) for c in coat)
    dark = (43, 36, 31)
    body_y = float(rng.uniform(43, 48))
    belly = float(rng.uniform(64, 70))
    rear, front = float(rng.uniform(19, 24)), float(rng.uniform(59, 64))
    # Common four-legged pose distribution, with independent joint offsets.
    for x in (rear + 6, rear + 12, front - 9, front - 3):
        knee = x + float(rng.uniform(-4, 4))
        hoof = knee + float(rng.uniform(-3, 3))
        line([(x, belly - 5), (knee, 76), (hoof, 85)], shade, 3.5)
        line([(hoof - 1, 85), (hoof + 3, 85)], dark, 2.5)
    # Tail length and endpoint vary for both species.
    tail_end = float(rng.uniform(66, 78))
    line([(rear + 2, body_y + 5), (rear - 5, 55), (rear - 7, tail_end)], shade, 2)
    ellipse((rear - 9, tail_end - 2, rear - 5, tail_end + 3), dark)
    ellipse((rear, body_y, front + 4, belly), coat)
    if cow:
        # Irregular patches are clipped to the torso, never to the landscape.
        for _ in range(int(rng.integers(3, 7))):
            x, y = rng.uniform(rear + 9, front - 4), rng.uniform(body_y + 6, belly - 6)
            radius = rng.uniform(3, 6)
            patch = [
                (
                    x + np.cos(a) * radius * rng.uniform(0.7, 1.3),
                    y + np.sin(a) * radius * rng.uniform(0.6, 1.0),
                )
                for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
            ]
            polygon(patch, shade)
        polygon(
            [
                (front - 4, body_y + 4),
                (front + 8, 39),
                (front + 15, 48),
                (front + 6, 64),
                (front - 1, belly - 2),
            ],
            coat,
        )
        ellipse((front + 4, 39, front + 24, 60), coat)
        ellipse((front + 13, 51, front + 27, 62), (188, 145, 126))
        for x in (front + 7, front + 18):
            polygon([(x, 42), (x - 2, 34), (x + 4, 41)], (216, 201, 161))
        ellipse((front - 1, 42, front + 8, 47), shade)
        eye = (front + 18, 47)
    else:
        # Raised neck, elongated muzzle, upright ears and flowing mane.
        polygon(
            [
                (front - 4, body_y + 7),
                (front + 5, 29),
                (front + 14, 28),
                (front + 15, 45),
                (front + 5, 65),
                (front - 2, belly - 3),
            ],
            coat,
        )
        polygon(
            [(front + 3, 30), (front + 7, 28), (front + 4, 49), (front - 3, 57), (front - 1, 42)],
            dark,
        )
        polygon(
            [
                (front + 8, 27),
                (front + 21, 31),
                (front + 29, 44),
                (front + 24, 50),
                (front + 13, 42),
            ],
            coat,
        )
        for x in (front + 9, front + 15):
            polygon([(x, 31), (x - 1, 21), (x + 5, 30)], shade)
        line([(front + 25, 44), (front + 28, 45)], shade, 3)
        eye = (front + 19, 35)
    ellipse((eye[0] - 1.2, eye[1] - 1.2, eye[0] + 1.2, eye[1] + 1.2), dark)
    if rng.random() < 0.5:
        canvas = ImageOps.mirror(canvas)
    angle = float(rng.uniform(-4, 4))
    canvas = canvas.rotate(angle, resample=Image.Resampling.BICUBIC)
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def render_background(source: Source, size: int) -> Image.Image:
    if source.kind != "background" or source.label not in ("grass", "beach"):
        raise ValueError(f"Unsupported background source: {source}")
    rng = np.random.default_rng(keyed_seed(source.seed, source.source_id))
    yy, xx = np.mgrid[0:size, 0:size]
    horizon = float(rng.uniform(0.30, 0.47)) * size
    lighting = float(rng.uniform(-17, 18))
    sky = np.array([153, 197, 218]) + rng.uniform(-14, 14, 3)
    ground = (
        np.array([77, 129, 61]) if source.label == "grass" else np.array([211, 190, 144])
    ) + rng.uniform(-17, 17, 3)
    landscape = horizon + np.sin(xx / size * rng.uniform(3, 10) + rng.uniform(0, 6)) * size * 0.025
    pixels = np.where((yy < landscape)[..., None], sky, ground)
    pixels += (yy / size)[..., None] * rng.uniform(-15, 10) + lighting
    pixels += rng.normal(0, rng.uniform(2, 6), (size, size, 1))
    if source.label == "beach":
        shoreline = landscape + size * rng.uniform(0.10, 0.20)
        sea = (yy >= landscape) & (yy < shoreline)
        pixels[sea] = np.array([87, 158, 174]) + rng.normal(0, 4, (sea.sum(), 1)) + lighting
        foam = (yy >= shoreline) & (yy < shoreline + size * 0.025)
        pixels[foam] = np.array([223, 226, 205]) + lighting
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    # Both backgrounds receive clouds; no object-dependent style or rendering paths.
    for _ in range(int(rng.integers(1, 4))):
        x, y = rng.integers(0, size), rng.integers(0, max(1, int(horizon * 0.7)))
        draw.ellipse((x, y, x + size // 4, y + max(1, size // 14)), fill=(209, 220, 217))
    for _ in range(size * 3):
        x, y = int(rng.integers(size)), int(rng.integers(int(size * 0.65), size))
        if source.label == "grass":
            color = tuple(int(c) for c in np.clip(ground + rng.uniform(-27, 25), 0, 255))
            draw.line((x, y, x + int(rng.integers(-1, 2)), y - int(rng.integers(1, 4))), fill=color)
        else:
            color = tuple(int(c) for c in np.clip(ground + rng.uniform(-20, 20), 0, 255))
            draw.point((x, y), fill=color)
    return image


def compose(foreground: Image.Image, background: Image.Image) -> Image.Image:
    if foreground.mode != "RGBA" or background.mode != "RGB":
        raise ValueError("Compositing requires RGBA foreground and RGB background")
    if foreground.size != background.size:
        raise ValueError("Foreground and background dimensions must match")
    return Image.alpha_composite(background.convert("RGBA"), foreground).convert("RGB")
