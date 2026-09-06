"""真实冻结 Qwen context 上的等步数 Expert 小试；只做离线 development 比较。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.g2c_memory_integration.vla import load_runtime, CHECKPOINT_SHA256
from experiments.memory_conditioning.conditioning import ObjectMemoryEncoder, MemoryBatch, MEMORY_SCHEMA
from robot_vla.model.qwen_context import QwenContext
from robot_vla.runtime.policy_runtime import _move_model_inputs
from robot_vla.training.flow_matching import sample_flow_training_target, masked_flow_mse


def validate_protocol(protocol):
    """运行参数绑定本次冻结问题，拒绝通过数据目录换 split 或训练预算。"""
    expected = dict(seeds=list(range(1000100,1000112)), train_seeds=list(range(1000100,1000108)),
        development_seeds=list(range(1000108,1000112)), train_steps_per_arm=32,
        batch_size=1, learning_rate=1e-5, sampling_seed=42)
    if any(protocol.get(key) != value for key,value in expected.items()):
        raise ValueError("数据协议与冻结的12个新seed/8:4划分/32步预算不一致")


def context_with_memory(context, encoder, features):
    token = encoder(MemoryBatch(features, torch.ones((1, 1), dtype=torch.bool, device=features.device)))
    return QwenContext(torch.cat((context.tokens, token.to(context.tokens.dtype)), 1),
        torch.cat((context.mask, torch.ones((1, 1), dtype=torch.bool, device=features.device)), 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = json.loads((args.data / "protocol.json").read_text())
    validate_protocol(protocol)
    collection = json.loads((args.data / "collection.json").read_text())
    records = collection["records"]
    if [r["seed"] for r in records] != protocol["seeds"]:
        raise ValueError("采集分母必须保留全部12个固定seed，且不得重排/重复/替换")
    denominator = {}
    for record in records:
        seed = record["seed"]
        result = record.get("result", {})
        capture = result.get("capture") or {}
        present = (args.data / str(seed) / "sample.npz").exists()
        if present != (capture.get("status") == "captured"):
            raise ValueError("样本文件与实际采集记录不一致")
        denominator[str(seed)] = dict(eligible=present, collection_status=record["status"],
            capture=capture, rejection_reasons=result.get("terminal_reasons", []), error=record.get("error"))
    split_paths = {split: [(s, args.data / str(s) / "sample.npz") for s in protocol[key]
        if (args.data / str(s) / "sample.npz").exists()]
        for split, key in (("train", "train_seeds"), ("development", "development_seeds"))}
    if any(not paths for paths in split_paths.values()):
        result = dict(status="inconclusive-insufficient-qualified-data", train_steps=0,
            counts={k:len(v) for k,v in split_paths.items()}, denominator=denominator)
        (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result)); return
    runtime, identity = load_runtime(args.checkpoint, args.model_cache)
    runtime.policy.eval()
    runtime.policy.adapter.requires_grad_(False)
    examples = {}
    hashes = {}
    data_metadata = {}
    for split, paths in split_paths.items():
        examples[split] = []
        for seed, path in paths:
            hashes[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
            metadata_path = path.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            if not metadata["snapshot"]["available"] or metadata["snapshot"]["schema"] != MEMORY_SCHEMA:
                raise ValueError("训练样本没有合格 Memory snapshot")
            data_metadata[str(seed)] = metadata
            hashes[str(seed)+"-metadata"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as data:
                if not np.allclose(data["memory_features"], metadata["snapshot"]["features"], rtol=1e-6, atol=1e-7):
                    raise ValueError("样本 Memory 数值与快照身份不一致")
                # 独立重算 commanded-target labels，首步使用显式重置参考。
                from robot_vla.adapters import ActionAdapter
                from robot_vla.contracts import RobotSpec
                reference = np.concatenate((data["initial_previous_command_q_rad"][None],
                    data["commanded_joint_target_rad"][:-1]))
                physical = np.concatenate((data["commanded_joint_target_rad"]-reference, np.ones((16,1))), axis=1).astype(np.float32)
                expected = ActionAdapter(RobotSpec()).normalize(physical, strict=True)
                if not np.allclose(expected, data["normalized_action"], rtol=0, atol=1e-6):
                    raise ValueError("动作标签不符合显式 commanded-target reference")
                encoded = runtime.processor_adapter.encode(data["rgb_external"], data["rgb_wrist"],
                    "pick the cube and place it in the target region")
                # no_grad 使缓存仍可被 Expert backward 保存；Qwen/Adapter 完全冻结。
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    context = runtime.policy.encode_context(_move_model_inputs(encoded.model_inputs, torch.device("cuda")))
                item = dict(seed=seed, context=context,
                    proprio=torch.tensor(runtime.proprio_normalizer.normalize(data["physical_proprio"])[None], device="cuda"),
                    action=torch.tensor(data["normalized_action"][None], device="cuda"),
                    features=torch.tensor(data["memory_features"][None], device="cuda"))
                if item["action"].shape != (1,16,8) or not all(torch.isfinite(item[k]).all()
                        for k in ("proprio", "action", "features")):
                    raise ValueError("数据 shape/finite 验证失败")
                examples[split].append(item)
    expert = runtime.policy.expert
    initial = {k:v.detach().cpu().clone() for k,v in expert.state_dict().items()}
    del runtime
    gc.collect(); torch.cuda.empty_cache()
    mask = torch.ones((1,16), dtype=torch.bool, device="cuda")
    results = {}

    def loss_for(item, encoder, generator):
        # 与已有 Runtime / Stage1 相同的 BF16 AMP；两条件共用，loss 仍累加 FP32。
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            context = item["context"] if encoder is None else context_with_memory(item["context"], encoder, item["features"])
            target = sample_flow_training_target(item["action"], mask, generator=generator)
            prediction = expert(context, item["proprio"], target.noisy_action, target.flow_time, mask)
            return masked_flow_mse(prediction, target.target_velocity, mask)

    def evaluate(encoder):
        expert.eval()
        values = {}
        with torch.no_grad():
            for item in examples["development"]:
                generator = torch.Generator(device="cuda").manual_seed(9000000 + item["seed"])
                values[str(item["seed"])] = float(torch.stack([
                    loss_for(item, encoder, generator) for _ in range(4)]).mean())
        return values

    for arm in ("no-memory", "memory"):
        expert.load_state_dict(initial, strict=True)
        torch.manual_seed(42)
        encoder = None if arm == "no-memory" else ObjectMemoryEncoder(expert.config.context_dim).cuda()
        parameters = list(expert.parameters()) + ([] if encoder is None else list(encoder.parameters()))
        optimizer = torch.optim.AdamW(parameters, lr=protocol["learning_rate"])
        generator = torch.Generator(device="cuda").manual_seed(42)
        before = evaluate(encoder)
        train_losses = []
        expert.train()
        for step in range(protocol["train_steps_per_arm"]):
            if time.monotonic()-started > 1000:
                raise TimeoutError("训练阶段达到保守时间上限；禁止不等预算效果结论")
            item = examples["train"][step % len(examples["train"])]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(item, encoder, generator)
            if not torch.isfinite(loss):
                raise ValueError("训练 loss 非有限")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        after = evaluate(encoder)
        masked = None if encoder is None else evaluate(None)
        payload = dict(format="memory-conditioning-m0/v1", upstream_sha256=CHECKPOINT_SHA256,
            memory_schema=MEMORY_SCHEMA, arm=arm, steps=len(train_losses), protocol=protocol,
            data_sha256=hashes, data_metadata=data_metadata,
            expert={k:v.detach().cpu() for k,v in expert.state_dict().items()},
            memory_encoder=None if encoder is None else {k:v.detach().cpu() for k,v in encoder.state_dict().items()})
        path = args.output / (arm + ".pt")
        torch.save(payload, path)
        del payload, optimizer
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        expert.load_state_dict(loaded["expert"], strict=True)
        if encoder is not None:
            encoder.load_state_dict(loaded["memory_encoder"], strict=True)
        del loaded
        reloaded = evaluate(encoder)
        if reloaded != after:
            raise RuntimeError("checkpoint 重载预测不一致")
        results[arm] = dict(steps=len(train_losses), train_loss_first=train_losses[0],
            train_loss_last=train_losses[-1], development_before=before, development_after=after,
            memory_removed_after_training=masked, checkpoint_bytes=path.stat().st_size,
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), strict_reload=True)
        (args.output / "progress.json").write_text(json.dumps(results, indent=2)+"\n")
        gc.collect(); torch.cuda.empty_cache()
    # 全部场景保留在资格分母；Flow MSE 仅在预定门槛合格的子集上有定义。
    result = dict(status="offline-qualified-subset-inconclusive-task-effect", identity=identity,
        denominator=denominator, planned_counts=dict(train=8, development=4),
        missing_handling="no replacement; unavailable labels have no numeric flow MSE; all rejections retained",
        effect_scope="conditional on qualified HOME Memory and valid teacher labels only",
        counts={k:len(v) for k,v in examples.items()}, results=results,
        elapsed_s=time.monotonic()-started, task_success_claim=False, protected_test=False)
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(status=result["status"], counts=result["counts"], elapsed_s=result["elapsed_s"])))


if __name__ == "__main__":
    main()
