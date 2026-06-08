"""
Unified profiling script for model complexity and model-level inference speed.

This script does not require a real dataset. It uses random input tensors to
measure:
    1. number of parameters,
    2. FLOPs with thop,
    3. throughput and latency under batched inference.

Default protocol:
    FLOPs input: 1 x 3 x 224 x 224
    Speed input: B x 3 x 224 x 224, where B = 256
    Warm-up iterations: 50
    Timed iterations: 200

Example:
    python tools/eval_models_speed_flops.py \
        --variants-json configs/eval/variants_pollen149_example.json \
        --batch-size 256 \
        --img-size 224 \
        --device cuda \
        --measure-flops \
        --warmup-iters 50 \
        --speed-iters 200 \
        --output-csv outputs/profiling_summary.csv
"""

import os
import sys
import json
import copy
import argparse
import gc
import time
import csv

import torch


def find_project_root():
    """
    Locate the classification project root.

    The project root is expected to contain:
        utils/config.py
        models/
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        file_dir,
        os.path.abspath(os.path.join(file_dir, "..")),
        os.path.abspath(os.path.join(file_dir, "..", "..")),
    ]

    for root in candidates:
        if (
            os.path.exists(os.path.join(root, "utils", "config.py"))
            and os.path.exists(os.path.join(root, "models"))
        ):
            return root

    # Fallback: assume this script is placed under tools/
    return os.path.abspath(os.path.join(file_dir, ".."))


CLASSIFICATION_ROOT = find_project_root()
sys.path.insert(0, CLASSIFICATION_ROOT)

from utils.config import get_config
from models import build_model


def parse_option():
    parser = argparse.ArgumentParser(
        "Evaluate model FLOPs, parameters, throughput, and latency with random tensors"
    )

    parser.add_argument(
        "--variants-json",
        type=str,
        required=True,
        help="Path to the variants JSON file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for speed profiling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use, e.g., cuda or cpu.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save profiling results as CSV.",
    )
    parser.add_argument(
        "--speed-iters",
        type=int,
        default=200,
        help="Number of timed iterations for speed profiling.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=50,
        help="Number of warm-up iterations before timing.",
    )
    parser.add_argument(
        "--measure-flops",
        action="store_true",
        help="Measure FLOPs using thop.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Input image size. The input tensor is square.",
    )

    # -------------------------------------------------------------------------
    # Compatibility arguments required by get_config(args).
    # These arguments are normally controlled by each variant in the JSON file.
    # -------------------------------------------------------------------------
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--opts", nargs="+", default=None)
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--cache-mode", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--amp-opt-level", type=str, default=None)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument("--fused-window-process", action="store_true")
    parser.add_argument("--fused-layernorm", action="store_true")
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()
    return args


def load_variants(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        variants = json.load(f)

    if not isinstance(variants, list):
        raise ValueError("variants-json must be a list of dictionaries.")

    required_keys = ["name", "cfg", "resume"]
    for index, variant in enumerate(variants):
        for key in required_keys:
            if key not in variant:
                raise ValueError(f"Variant index {index} is missing required key: {key}")

        variant.setdefault("opts", [])
        variant.setdefault("freq_override", "none")

        if not isinstance(variant["opts"], list):
            raise ValueError(f"Variant {variant['name']} opts must be a list.")

    return variants


def apply_freq_override(config, freq_override):
    """
    Optional compatibility handling for the JSON field freq_override.

    Supported values:
        none/config: do nothing
        disable    : force-disable frequency-domain regularization if present

    In this repository:
        MODEL.PARTIAL_HEAT corresponds to Partial Heat Routing (PHR).
        MODEL.FP corresponds to Frequency-domain Regularization (FDR).
    """
    mode = str(freq_override).lower()

    if mode in ["none", "config"]:
        return config

    if mode != "disable":
        print(f"[Warning] Unknown freq_override={freq_override}. Skip explicit handling.")
        return config

    config.defrost()

    if hasattr(config.MODEL, "FP") and hasattr(config.MODEL.FP, "ENABLE"):
        config.MODEL.FP.ENABLE = False

    config.freeze()
    return config


def make_config_for_variant(base_args, variant, img_size):
    """
    Create a config object for a specific model variant.

    This script is used only for profiling, so eval and throughput modes are
    forced on to make model construction consistent with inference profiling.
    """
    tmp_args = copy.deepcopy(base_args)

    tmp_args.cfg = variant["cfg"]
    tmp_args.resume = variant["resume"]
    tmp_args.opts = variant.get("opts", [])

    # Force inference/profiling mode.
    tmp_args.eval = True
    tmp_args.throughput = True

    config = get_config(tmp_args)

    config.defrost()

    if hasattr(config.DATA, "IMG_SIZE"):
        config.DATA.IMG_SIZE = img_size

    config.freeze()

    config = apply_freq_override(config, variant.get("freq_override", "none"))
    return config


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Checkpoint loaded: {ckpt_path}")
    print(msg)

    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "params": total,
        "trainable_params": trainable,
        "params_m": total / 1e6,
        "trainable_params_m": trainable / 1e6,
    }


def measure_flops(model, img_size, device):
    """
    Measure FLOPs using thop.

    The input tensor size is:
        1 x 3 x img_size x img_size
    """
    try:
        from thop import profile
    except ImportError:
        print("[Warning] thop is not installed. FLOPs will be set to None.")
        print("Install it with: pip install thop")
        return None

    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    try:
        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops_g = flops / 1e9
    except Exception as exc:
        print(f"[Warning] thop FLOPs measurement failed: {exc}")
        flops_g = None

    del dummy

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return flops_g


def clear_model_hooks(model):
    """
    Remove possible residual hooks and attributes created by profiling tools.
    """
    for module in model.modules():
        if hasattr(module, "_forward_hooks"):
            module._forward_hooks.clear()
        if hasattr(module, "_forward_pre_hooks"):
            module._forward_pre_hooks.clear()
        if hasattr(module, "_backward_hooks"):
            module._backward_hooks.clear()

        for attr in ["total_ops", "total_params"]:
            if hasattr(module, attr):
                delattr(module, attr)


@torch.no_grad()
def measure_speed(model, batch_size, img_size, device, warmup_iters, speed_iters):
    """
    Measure batched model-level inference speed using random input tensors.

    The input tensor size is:
        batch_size x 3 x img_size x img_size

    Throughput:
        images / second

    Latency:
        average milliseconds per image under batched inference
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    for _ in range(warmup_iters):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(speed_iters):
            _ = model(dummy)
        end.record()

        torch.cuda.synchronize()
        total_time = start.elapsed_time(end) / 1000.0
    else:
        start_time = time.perf_counter()
        for _ in range(speed_iters):
            _ = model(dummy)
        total_time = time.perf_counter() - start_time

    total_images = batch_size * speed_iters
    throughput = total_images / max(total_time, 1e-12)
    latency_ms_per_image = 1000.0 * total_time / max(total_images, 1)

    del dummy

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return throughput, latency_ms_per_image


def evaluate_one_variant(base_args, variant, device):
    print("=" * 100)
    print(f"Experiment : {variant['name']}")
    print(f"Config     : {variant['cfg']}")
    print(f"Checkpoint : {variant['resume']}")
    print(f"Opts       : {variant.get('opts', [])}")
    print("=" * 100)

    config = make_config_for_variant(base_args, variant, base_args.img_size)

    model = build_model(config)
    model = load_checkpoint(model, variant["resume"])
    model.to(device)
    model.eval()

    params_info = count_params(model)
    print(f"Params: {params_info['params_m']:.4f} M")

    flops_g = None
    if base_args.measure_flops:
        flops_g = measure_flops(model, base_args.img_size, device)
        clear_model_hooks(model)

        if flops_g is not None:
            print(f"FLOPs: {flops_g:.4f} G")
        else:
            print("FLOPs: N/A")

    throughput, latency_ms = measure_speed(
        model=model,
        batch_size=base_args.batch_size,
        img_size=base_args.img_size,
        device=device,
        warmup_iters=base_args.warmup_iters,
        speed_iters=base_args.speed_iters,
    )

    print(f"Throughput: {throughput:.2f} images/s")
    print(f"Latency: {latency_ms:.4f} ms/image")

    result = {
        "name": variant["name"],
        "params_m": params_info["params_m"],
        "trainable_params_m": params_info["trainable_params_m"],
        "flops_g": flops_g,
        "throughput": throughput,
        "latency_ms_per_image": latency_ms,
        "batch_size": base_args.batch_size,
        "img_size": base_args.img_size,
        "warmup_iters": base_args.warmup_iters,
        "speed_iters": base_args.speed_iters,
        "cfg": variant["cfg"],
        "resume": variant["resume"],
        "freq_override": variant.get("freq_override", "none"),
    }

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def save_csv(results, csv_path):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    fieldnames = [
        "name",
        "params_m",
        "trainable_params_m",
        "flops_g",
        "throughput",
        "latency_ms_per_image",
        "batch_size",
        "img_size",
        "warmup_iters",
        "speed_iters",
        "cfg",
        "resume",
        "freq_override",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"Saved CSV to: {csv_path}")


def print_summary(results):
    print("\n" + "=" * 120)
    print("Final summary")
    print("=" * 120)

    header = (
        f"{'Experiment':<18} "
        f"{'Params(M)':>10} "
        f"{'FLOPs(G)':>10} "
        f"{'Throughput':>14} "
        f"{'ms/image':>10} "
        f"{'B':>6} "
        f"{'Img':>6}"
    )
    print(header)
    print("-" * 120)

    for result in results:
        flops_str = f"{result['flops_g']:.4f}" if result["flops_g"] is not None else "N/A"
        print(
            f"{result['name']:<18} "
            f"{result['params_m']:>10.4f} "
            f"{flops_str:>10} "
            f"{result['throughput']:>14.2f} "
            f"{result['latency_ms_per_image']:>10.4f} "
            f"{result['batch_size']:>6} "
            f"{result['img_size']:>6}"
        )

    print("=" * 120)


def main():
    args = parse_option()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Project root: {CLASSIFICATION_ROOT}")
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Warm-up iterations: {args.warmup_iters}")
    print(f"Timed iterations: {args.speed_iters}")

    variants = load_variants(args.variants_json)
    print(f"Loaded {len(variants)} variants from: {args.variants_json}")

    all_results = []

    for variant in variants:
        try:
            result = evaluate_one_variant(args, variant, device)
            all_results.append(result)
        except Exception as exc:
            print(f"[Error] Failed for {variant['name']}: {exc}")

    print_summary(all_results)

    if args.output_csv:
        save_csv(all_results, args.output_csv)


if __name__ == "__main__":
    main()
