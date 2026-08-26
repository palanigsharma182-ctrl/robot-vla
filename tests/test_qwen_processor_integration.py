import numpy as np
import pytest

pytest.importorskip("transformers")


def test_qwen_processor_encodes_bounded_external_then_wrist_tokens(
    qwen_processor_adapter,
) -> None:
    processor_adapter = qwen_processor_adapter
    external = np.full((512, 512, 3), 32, dtype=np.uint8)
    wrist = np.full((480, 640, 3), 96, dtype=np.uint8)
    instruction = "Pick up the red cube."

    prompt = processor_adapter.format_prompt(external, wrist, instruction)
    batch = processor_adapter.encode(external, wrist, instruction)

    assert "<|im_start|>assistant" not in prompt
    assert prompt.index("External/front camera:") < prompt.index("Wrist camera:")
    assert prompt.index("Wrist camera:") < prompt.index("Robot instruction:")
    assert prompt.endswith("<|im_end|>\n")
    assert batch.visual_tokens_per_image == ((256, 234),)
    assert batch.model_inputs["input_ids"].shape[0] == 1
    assert batch.model_inputs["attention_mask"].dtype.is_floating_point is False
    assert batch.model_inputs["attention_mask"].dtype == pytest.importorskip("torch").bool
    assert set(batch.model_inputs) == {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "image_grid_thw",
    }


def test_qwen_processor_batches_without_silent_instruction_truncation(
    qwen_processor_adapter,
) -> None:
    processor_adapter = qwen_processor_adapter
    external = [
        np.zeros((512, 512, 3), dtype=np.uint8),
        np.zeros((256, 256, 3), dtype=np.uint8),
    ]
    wrist = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((256, 256, 3), dtype=np.uint8),
    ]
    batch = processor_adapter.encode_batch(
        external,
        wrist,
        ["Pick the cube.", "Pick the cube and place it in the target region."],
    )

    assert batch.visual_tokens_per_image == ((256, 234), (64, 64))
    assert batch.model_inputs["input_ids"].shape[0] == 2
    assert len(batch.context_lengths) == 2

    with pytest.raises(ValueError, match="instruction token 数"):
        processor_adapter.encode(
            external[0],
            wrist[0],
            "robot " * 65,
        )
