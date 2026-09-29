"""The built-in prompt presets, seeded into the Prompt library by a migration.

Every one leads with the same geometry clause: the labels are copied onto the
result unchanged, so the one thing a prompt must never do is move, add or
remove an annotated object. The validator checks that afterwards, but a prompt
that asks for it up front gets rejected far less often.

To add a preset, append to ``PRESETS`` and add a data migration that calls
``seed_presets`` — or just add it in the Prompt library tab.
"""

_PRESERVE = (
    "Preserve the exact geometry of the input image. "
    "Do not move, resize, add, remove, replace, or substantially modify any person, "
    "hand, weapon, PPE item, or other foreground object. "
    "Do not change camera position, perspective, object locations, body pose, hand "
    "pose, or object boundaries. Only modify the visual appearance of the scene."
)

PRESETS = {
    "night_cctv": (
        "Nighttime CCTV footage",
        f"{_PRESERVE}\n\n"
        "Transform the image into realistic nighttime CCTV footage with reduced "
        "illumination, realistic sensor noise, slightly reduced dynamic range, subtle "
        "compression artifacts, and realistic surveillance-camera color characteristics.\n\n"
        "The spatial positions and shapes of all foreground objects must remain unchanged.",
    ),
    "compression": (
        "Heavily compressed CCTV stream",
        "Preserve all scene geometry and object locations exactly. Do not add, remove, "
        "move, resize, or replace objects.\n\n"
        "Make the image resemble heavily compressed CCTV footage. Apply realistic "
        "H.264-like compression degradation, block artifacts, reduced fine detail, "
        "slight chroma degradation, and mild sensor noise.\n\n"
        "Do not change scene composition or object geometry.",
    ),
    "low_light": (
        "Low-light surveillance",
        "Preserve all objects, people, poses, bounding-box geometry, camera position, "
        "and scene composition. Only change the lighting.\n\n"
        "Convert the image to realistic low-light surveillance footage with darker "
        "ambient illumination, realistic shadows, reduced contrast, sensor noise, and a "
        "subtle color shift.\n\n"
        "Do not move, add, remove, resize, or replace anything.",
    ),
    "motion_blur": (
        "Mild CCTV motion blur",
        "Preserve the complete scene geometry. Do not move, add, remove, resize, or "
        "replace any object.\n\n"
        "Apply realistic mild CCTV motion blur and exposure characteristics while "
        "maintaining the original locations and shapes of all objects.\n\n"
        "The result should remain recognizable and suitable for object-detection training.",
    ),
    "fluorescent_indoor": (
        "Indoor fluorescent lighting",
        f"{_PRESERVE}\n\n"
        "Relight the scene as if it were filmed indoors under cool fluorescent tube "
        "lighting: a slight green-blue color cast, flat and even illumination, soft "
        "shadows, mildly reduced saturation, and the fine sensor noise of an indoor "
        "security camera.\n\n"
        "The spatial positions and shapes of all foreground objects must remain unchanged.",
    ),
}


def seed_presets(AugPrompt) -> None:
    """Create any missing built-in preset. Never overwrites an edited one."""
    for name, (description, text) in PRESETS.items():
        AugPrompt.objects.get_or_create(
            name=name, defaults={"text": text, "description": description, "builtin": True},
        )
