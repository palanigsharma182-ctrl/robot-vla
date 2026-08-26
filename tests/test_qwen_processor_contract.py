import numpy as np
import pytest

from robot_vla.model.qwen_processor import (
    SYSTEM_PROMPT,
    QwenProcessorConfig,
    build_qwen_conversation,
)


def test_visual_pixel_budget_is_derived_from_patch_and_merge() -> None:
    config = QwenProcessorConfig()

    assert config.pixels_per_visual_token == 1024
    assert config.min_pixels == 65_536
    assert config.max_pixels == 262_144


def test_conversation_fixes_camera_roles_and_order() -> None:
    external = np.full((16, 20, 3), 32, dtype=np.uint8)
    wrist = np.full((12, 12, 3), 96, dtype=np.uint8)

    conversation = build_qwen_conversation(
        external,
        wrist,
        "Pick up the red cube.",
    )

    assert conversation[0] == {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    }
    user_content = conversation[1]["content"]
    assert user_content[0] == {"type": "text", "text": "External/front camera:\n"}
    assert np.all(user_content[1]["image"] == 32)
    assert user_content[2] == {"type": "text", "text": "\n\nWrist camera:\n"}
    assert np.all(user_content[3]["image"] == 96)
    assert user_content[4]["text"].endswith("Pick up the red cube.")


def test_conversation_rejects_missing_or_non_uint8_image() -> None:
    valid = np.zeros((16, 16, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="rgb_external"):
        build_qwen_conversation(np.zeros((16, 16, 3), dtype=np.float32), valid, "pick")
    with pytest.raises(ValueError, match="rgb_wrist"):
        build_qwen_conversation(valid, np.zeros((16, 16), dtype=np.uint8), "pick")
    with pytest.raises(ValueError, match="instruction"):
        build_qwen_conversation(valid, valid, " ")
