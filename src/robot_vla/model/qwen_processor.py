"""qwen-vla-prompt/v1 与固定 Qwen3.5 双图 Processor 的适配边界。"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_vla.contracts import (
    OBSERVATION_HISTORY_LENGTH,
    PROMPT_VERSION,
    PROMPT_VERSION_OBSERVATION_V2,
    QWEN_MODEL_ID,
    QWEN_REVISION,
)

SYSTEM_PROMPT = (
    "You control a 7-DoF robot arm with a two-finger gripper.\n"
    "Encode the observation and instruction for continuous robot control."
)
SYSTEM_PROMPT_V2 = (
    "You control a 7-DoF robot arm with a two-finger gripper.\n"
    "The observation contains four consecutive control steps ordered oldest to newest.\n"
    "Encode the synchronized camera history and instruction for continuous robot control."
)
VLA_IMAGE_TIME_INDICES = "vla_image_time_indices"
VLA_CONTEXT_VALID_MASK = "vla_context_valid_mask"


@dataclass(frozen=True)
class QwenProcessorConfig:
    model_id: str = QWEN_MODEL_ID
    revision: str = QWEN_REVISION
    prompt_version: str = PROMPT_VERSION
    max_instruction_tokens: int = 64
    min_visual_tokens_per_image: int = 64
    max_visual_tokens_per_image: int = 256
    patch_size: int = 16
    merge_size: int = 2

    def __post_init__(self) -> None:
        if self.model_id != QWEN_MODEL_ID or self.revision != QWEN_REVISION:
            raise ValueError("Qwen model ID 和 revision 必须与 qwen-vla-v0.1 身份一致")
        if self.prompt_version not in (
            PROMPT_VERSION,
            PROMPT_VERSION_OBSERVATION_V2,
        ):
            raise ValueError("Prompt version 必须是已注册的 Observation V1/V2 版本")
        if self.max_instruction_tokens <= 0:
            raise ValueError("max_instruction_tokens 必须为正数")
        if not 0 < self.min_visual_tokens_per_image <= self.max_visual_tokens_per_image:
            raise ValueError("visual token 上下限无效")
        if self.patch_size <= 0 or self.merge_size <= 0:
            raise ValueError("patch_size 和 merge_size 必须为正数")

    @property
    def pixels_per_visual_token(self) -> int:
        return (self.patch_size * self.merge_size) ** 2

    @property
    def min_pixels(self) -> int:
        return self.min_visual_tokens_per_image * self.pixels_per_visual_token

    @property
    def max_pixels(self) -> int:
        return self.max_visual_tokens_per_image * self.pixels_per_visual_token


@dataclass(frozen=True)
class ProcessedObservationBatch:
    model_inputs: dict[str, Any]
    visual_tokens_per_image: tuple[tuple[int, ...], ...]
    context_lengths: tuple[int, ...]


def _validate_rgb(image: Any, name: str) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"{name} 应为 [H,W,3] RGB，实际为 {value.shape}")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} 的 H/W 必须为正数")
    if value.dtype != np.uint8:
        raise ValueError(f"{name} dtype 应为 uint8，实际为 {value.dtype}")
    return np.ascontiguousarray(value)


def build_qwen_conversation(
    rgb_external: Any,
    rgb_wrist: Any,
    instruction: str,
) -> list[dict[str, Any]]:
    """按固定 external -> wrist 顺序构造一次非生成式多模态对话。"""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction 必须是非空字符串")
    external = _validate_rgb(rgb_external, "rgb_external")
    wrist = _validate_rgb(rgb_wrist, "rgb_wrist")
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "External/front camera:\n"},
                {"type": "image", "image": external},
                {"type": "text", "text": "\n\nWrist camera:\n"},
                {"type": "image", "image": wrist},
                {"type": "text", "text": f"\n\nRobot instruction:\n{instruction}"},
            ],
        },
    ]


def build_qwen_history_conversation(
    rgb_external_history: Any,
    rgb_wrist_history: Any,
    history_valid: Any,
    instruction: str,
) -> list[dict[str, Any]]:
    """按 ``t-3→t``、每步 ``external→wrist`` 的固定顺序构造八图输入。"""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction 必须是非空字符串")
    external = np.asarray(rgb_external_history)
    wrist = np.asarray(rgb_wrist_history)
    valid = np.asarray(history_valid)
    if external.ndim != 4 or external.shape[0] != OBSERVATION_HISTORY_LENGTH:
        raise ValueError("rgb_external_history 必须是 [4,H,W,3]")
    if wrist.ndim != 4 or wrist.shape[0] != OBSERVATION_HISTORY_LENGTH:
        raise ValueError("rgb_wrist_history 必须是 [4,H,W,3]")
    if valid.shape != (OBSERVATION_HISTORY_LENGTH,) or valid.dtype != np.bool_:
        raise ValueError("history_valid 必须是 bool [4]")
    content: list[dict[str, Any]] = []
    for time_index in range(OBSERVATION_HISTORY_LENGTH):
        external_image = _validate_rgb(
            external[time_index],
            f"rgb_external_history[{time_index}]",
        )
        wrist_image = _validate_rgb(
            wrist[time_index],
            f"rgb_wrist_history[{time_index}]",
        )
        status = "valid" if bool(valid[time_index]) else "padding-invalid"
        relative_step = time_index - (OBSERVATION_HISTORY_LENGTH - 1)
        content.extend(
            (
                {
                    "type": "text",
                    "text": (
                        f"History step {relative_step:+d} ({status}), external/front camera:\n"
                    ),
                },
                {"type": "image", "image": external_image},
                {"type": "text", "text": "\nWrist camera:\n"},
                {"type": "image", "image": wrist_image},
                {"type": "text", "text": "\n\n"},
            )
        )
    content.append({"type": "text", "text": f"Robot instruction:\n{instruction}"})
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT_V2}],
        },
        {"role": "user", "content": content},
    ]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _contiguous_run_lengths(positions: np.ndarray) -> tuple[int, ...]:
    if positions.size == 0:
        return ()
    split_indices = np.flatnonzero(np.diff(positions) != 1) + 1
    return tuple(len(run) for run in np.split(positions, split_indices))


def _contiguous_runs(positions: np.ndarray) -> tuple[np.ndarray, ...]:
    if positions.size == 0:
        return ()
    split_indices = np.flatnonzero(np.diff(positions) != 1) + 1
    return tuple(np.split(positions, split_indices))


class QwenVLAProcessorAdapter:
    """把双图和指令编码为 Qwen 模型输入，并检查所有版本化预算。"""

    def __init__(self, processor: Any, config: QwenProcessorConfig | None = None) -> None:
        self.processor = processor
        self.config = config or QwenProcessorConfig()
        if type(processor).__name__ != "Qwen3VLProcessor":
            raise ValueError(
                f"固定 revision 应使用 Qwen3VLProcessor，实际为 {type(processor).__name__}"
            )
        image_processor = processor.image_processor
        if int(image_processor.patch_size) != self.config.patch_size:
            raise ValueError("Qwen image patch_size 与固定配置不一致")
        if int(image_processor.merge_size) != self.config.merge_size:
            raise ValueError("Qwen image merge_size 与固定配置不一致")
        size = image_processor.size
        if int(size.shortest_edge) != self.config.min_pixels:
            raise ValueError("Qwen image min_pixels 与固定视觉预算不一致")
        if int(size.longest_edge) != self.config.max_pixels:
            raise ValueError("Qwen image max_pixels 与固定视觉预算不一致")

    @classmethod
    def from_pretrained(
        cls,
        *,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        hf_endpoint: str = "https://hf-mirror.com",
        config: QwenProcessorConfig | None = None,
    ) -> QwenVLAProcessorAdapter:
        resolved = config or QwenProcessorConfig()
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError("QwenVLAProcessorAdapter 需要安装 transformers") from exc
        processor = AutoProcessor.from_pretrained(
            resolved.model_id,
            revision=resolved.revision,
            trust_remote_code=False,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            min_pixels=resolved.min_pixels,
            max_pixels=resolved.max_pixels,
        )
        return cls(processor, resolved)

    def _validate_instruction(self, instruction: str) -> None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction 必须是非空字符串")
        tokens = self.processor.tokenizer(
            instruction,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if len(tokens) > self.config.max_instruction_tokens:
            raise ValueError(
                f"instruction token 数 {len(tokens)} 超过上限 "
                f"{self.config.max_instruction_tokens}"
            )

    def format_prompt(self, rgb_external: Any, rgb_wrist: Any, instruction: str) -> str:
        self._validate_instruction(instruction)
        conversation = build_qwen_conversation(rgb_external, rgb_wrist, instruction)
        prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        if "<|im_start|>assistant" in prompt:
            raise RuntimeError("qwen-vla-prompt/v1 不能包含 assistant generation header")
        return str(prompt)

    def encode(
        self,
        rgb_external: Any,
        rgb_wrist: Any,
        instruction: str,
    ) -> ProcessedObservationBatch:
        return self.encode_batch([rgb_external], [rgb_wrist], [instruction])

    def encode_batch(
        self,
        rgb_external: Sequence[Any],
        rgb_wrist: Sequence[Any],
        instructions: Sequence[str],
    ) -> ProcessedObservationBatch:
        batch_size = len(instructions)
        if batch_size == 0:
            raise ValueError("不能编码空 batch")
        if len(rgb_external) != batch_size or len(rgb_wrist) != batch_size:
            raise ValueError("双相机和 instruction batch size 必须相同")
        conversations = []
        for external, wrist, instruction in zip(
            rgb_external,
            rgb_wrist,
            instructions,
            strict=True,
        ):
            self._validate_instruction(instruction)
            conversations.append(build_qwen_conversation(external, wrist, instruction))

        encoded = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        model_inputs = dict(encoded)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].bool()
        visual_tokens = self._validate_encoded(model_inputs, batch_size)
        context_lengths = tuple(
            int(value)
            for value in _as_numpy(model_inputs["attention_mask"]).sum(axis=1).tolist()
        )
        return ProcessedObservationBatch(
            model_inputs=model_inputs,
            visual_tokens_per_image=visual_tokens,
            context_lengths=context_lengths,
        )

    def encode_history(
        self,
        rgb_external_history: Any,
        rgb_wrist_history: Any,
        history_valid: Any,
        instruction: str,
    ) -> ProcessedObservationBatch:
        return self.encode_history_batch(
            [rgb_external_history],
            [rgb_wrist_history],
            [history_valid],
            [instruction],
        )

    def encode_history_batch(
        self,
        rgb_external_history: Sequence[Any],
        rgb_wrist_history: Sequence[Any],
        history_valid: Sequence[Any],
        instructions: Sequence[str],
    ) -> ProcessedObservationBatch:
        batch_size = len(instructions)
        if batch_size == 0:
            raise ValueError("不能编码空 history batch")
        if not (
            len(rgb_external_history)
            == len(rgb_wrist_history)
            == len(history_valid)
            == batch_size
        ):
            raise ValueError("双相机 history、valid 和 instruction batch size 必须相同")
        conversations = []
        resolved_valid: list[np.ndarray] = []
        for external, wrist, valid, instruction in zip(
            rgb_external_history,
            rgb_wrist_history,
            history_valid,
            instructions,
            strict=True,
        ):
            self._validate_instruction(instruction)
            valid_array = np.asarray(valid)
            conversations.append(
                build_qwen_history_conversation(external, wrist, valid_array, instruction)
            )
            resolved_valid.append(valid_array.copy())
        encoded = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        model_inputs = dict(encoded)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].bool()
        image_time_indices = tuple(
            time_index
            for time_index in range(OBSERVATION_HISTORY_LENGTH)
            for _camera_index in range(2)
        )
        visual_tokens = self._validate_encoded(
            model_inputs,
            batch_size,
            images_per_sample=OBSERVATION_HISTORY_LENGTH * 2,
            image_time_indices=image_time_indices,
            history_valid=resolved_valid,
        )
        context_lengths = tuple(
            int(value)
            for value in _as_numpy(model_inputs["attention_mask"]).sum(axis=1).tolist()
        )
        return ProcessedObservationBatch(
            model_inputs=model_inputs,
            visual_tokens_per_image=visual_tokens,
            context_lengths=context_lengths,
        )

    def _validate_encoded(
        self,
        model_inputs: dict[str, Any],
        batch_size: int,
        *,
        images_per_sample: int = 2,
        image_time_indices: tuple[int, ...] | None = None,
        history_valid: Sequence[np.ndarray] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        required = {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        }
        missing = required.difference(model_inputs)
        if missing:
            raise RuntimeError(f"Qwen Processor 输出缺少字段: {sorted(missing)}")

        input_ids = _as_numpy(model_inputs["input_ids"])
        attention_mask = _as_numpy(model_inputs["attention_mask"])
        grids = _as_numpy(model_inputs["image_grid_thw"])
        if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
            raise RuntimeError(f"input_ids 应为 [B,N]，实际为 {input_ids.shape}")
        if attention_mask.shape != input_ids.shape:
            raise RuntimeError("attention_mask shape 必须与 input_ids 相同")
        if grids.shape != (batch_size * images_per_sample, 3):
            raise RuntimeError(
                "image_grid_thw 应为 "
                f"[{batch_size * images_per_sample},3]，实际为 {grids.shape}"
            )
        if image_time_indices is not None and len(image_time_indices) != images_per_sample:
            raise RuntimeError("image_time_indices 数量必须与每样本图像数一致")
        if (image_time_indices is None) != (history_valid is None):
            raise RuntimeError("history time index 与 validity 必须同时提供")

        merge_area = self.config.merge_size**2
        flat_visual_tokens: list[int] = []
        for grid in grids:
            patch_tokens = int(np.prod(grid, dtype=np.int64))
            if patch_tokens % merge_area != 0:
                raise RuntimeError("image_grid_thw 不能被 merge_size 整除")
            count = patch_tokens // merge_area
            if not (
                self.config.min_visual_tokens_per_image
                <= count
                <= self.config.max_visual_tokens_per_image
            ):
                raise RuntimeError(f"单图 visual token 数 {count} 超出固定预算")
            flat_visual_tokens.append(count)

        image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        per_sample: list[tuple[int, ...]] = []
        auxiliary_time = np.full(input_ids.shape, -1, dtype=np.int64)
        context_valid = attention_mask.astype(np.bool_, copy=True)
        for batch_index in range(batch_size):
            start = batch_index * images_per_sample
            expected = tuple(flat_visual_tokens[start : start + images_per_sample])
            positions = np.flatnonzero(input_ids[batch_index] == image_token_id)
            runs = _contiguous_runs(positions)
            actual_runs = tuple(len(run) for run in runs)
            if actual_runs != expected:
                raise RuntimeError(
                    f"样本 {batch_index} 的图像 token span 应为 {expected}，实际为 {actual_runs}"
                )
            if image_time_indices is not None:
                assert history_valid is not None
                valid = np.asarray(history_valid[batch_index])
                if valid.shape != (OBSERVATION_HISTORY_LENGTH,) or valid.dtype != np.bool_:
                    raise RuntimeError("history_valid 必须是 bool [4]")
                for run, time_index in zip(runs, image_time_indices, strict=True):
                    auxiliary_time[batch_index, run] = time_index
                    if not bool(valid[time_index]):
                        context_valid[batch_index, run] = False
            per_sample.append(expected)
        if image_time_indices is not None:
            template = model_inputs["input_ids"]
            if hasattr(template, "new_tensor"):
                model_inputs[VLA_IMAGE_TIME_INDICES] = template.new_tensor(auxiliary_time)
                valid_tensor = template.new_tensor(
                    context_valid,
                    dtype=getattr(model_inputs["attention_mask"], "dtype", None),
                ).bool()
                model_inputs[VLA_CONTEXT_VALID_MASK] = valid_tensor
                # padding 图像仍占固定八图 layout，但不能作为 Qwen K/V 影响有效文本或图像。
                model_inputs["attention_mask"] = (
                    model_inputs["attention_mask"].bool() & valid_tensor
                )
            else:
                model_inputs[VLA_IMAGE_TIME_INDICES] = auxiliary_time
                model_inputs[VLA_CONTEXT_VALID_MASK] = context_valid
                model_inputs["attention_mask"] = (
                    np.asarray(model_inputs["attention_mask"], dtype=np.bool_)
                    & context_valid
                )
        return tuple(per_sample)
