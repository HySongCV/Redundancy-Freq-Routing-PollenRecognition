import os
import sys
import json
import copy
import argparse
import torch
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_ROOT = os.path.abspath(os.path.join(FILE_DIR, "..", ".."))
sys.path.insert(0, CLASSIFICATION_ROOT)

from utils.config import get_config
from data import build_loader
from models import build_model


def parse_option():
    parser = argparse.ArgumentParser("Evaluate multiple vHeat variants from JSON")

    parser.add_argument("--variants-json", type=str, required=True,
                        help="Path to variants JSON file.")

    parser.add_argument("--data-path", type=str, default=None,
                        help="Override dataset path for all variants.")

    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for all variants.")

    parser.add_argument("--eval-set", type=str, default="test", choices=["val", "test"],
                        help="Evaluate validation set or test set.")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use.")

    parser.add_argument("--output-csv", type=str, default=None,
                        help="Optional path to save summary results as CSV.")

    parser.add_argument("--speed-iters", type=int, default=50,
                    help="Number of batches used for speed test.")

    parser.add_argument("--warmup-iters", type=int, default=10,
                        help="Number of warmup batches before speed test.")
    
    parser.add_argument("--measure-flops", action="store_true",
                        help="Measure FLOPs if thop is installed.")

    parser.add_argument("--local_rank", type=int, default=0)

    # ===== compatibility arguments required by get_config(args) =====
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

    args = parser.parse_args()
    return args


def load_variants(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        variants = json.load(f)

    if not isinstance(variants, list):
        raise ValueError(
            "variants-json must be a list, for example: "
            "[{'name': 'Original', 'cfg': '...', 'resume': '...', 'opts': [...]}]"
        )

    required_keys = ["name", "cfg", "resume"]
    for i, v in enumerate(variants):
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Variant index {i} missing required key: {key}")

        if "opts" not in v:
            v["opts"] = []

        if "freq_override" not in v:
            v["freq_override"] = "none"

        if not isinstance(v["opts"], list):
            raise ValueError(f"Variant {v['name']} opts must be a list.")

    return variants


def apply_freq_override(config, freq_override):
    """
    Compatible handling for your JSON field:
      - disable: force disable frequency branch if related config keys exist
      - config : keep config / opts setting
      - none   : do nothing

    This function is defensive because different versions of config may define
    FP / frequency-related keys differently.
    """
    if freq_override is None:
        return config

    mode = str(freq_override).lower()

    if mode in ["none", "config"]:
        return config

    if mode != "disable":
        print(f"[Warning] Unknown freq_override={freq_override}, skip explicit handling.")
        return config

    config.defrost()

    # Your current config seems to use MODEL.FP.ENABLE.
    if hasattr(config.MODEL, "FP"):
        if hasattr(config.MODEL.FP, "ENABLE"):
            config.MODEL.FP.ENABLE = False

    # Defensive compatibility for other possible names.
    if hasattr(config.MODEL, "FREQ"):
        if hasattr(config.MODEL.FREQ, "ENABLE"):
            config.MODEL.FREQ.ENABLE = False

    if hasattr(config.MODEL, "FREQUENCY"):
        if hasattr(config.MODEL.FREQUENCY, "ENABLE"):
            config.MODEL.FREQUENCY.ENABLE = False

    config.freeze()
    return config


def make_config_for_variant(base_args, variant):
    """
    Each variant has its own cfg/resume/opts/freq_override.
    The key point is:
      tmp_args.cfg = variant["cfg"]
      tmp_args.resume = variant["resume"]
      tmp_args.opts = variant["opts"]
    """
    tmp_args = copy.deepcopy(base_args)

    tmp_args.cfg = variant["cfg"]
    tmp_args.resume = variant["resume"]
    tmp_args.opts = variant.get("opts", [])

    config = get_config(tmp_args)

    config.defrost()

    if base_args.data_path is not None:
        config.DATA.DATA_PATH = base_args.data_path

    if base_args.batch_size is not None:
        config.DATA.BATCH_SIZE = base_args.batch_size

    config.freeze()

    config = apply_freq_override(config, variant.get("freq_override", "none"))

    return config


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

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


def measure_flops(model, input_size=(1, 3, 224, 224), device="cuda"):
    """
    Measure FLOPs with thop.

    Important:
    Use a deepcopy of the model to avoid thop hooks polluting the original model.
    """
    try:
        from thop import profile
    except ImportError:
        print("[Warning] thop is not installed. FLOPs will be set to None.")
        print("You can install it with: pip install thop")
        return None

    import copy
    import gc

    # 用副本测 FLOPs，避免 thop hook 留在原模型上
    model_for_flops = copy.deepcopy(model)
    model_for_flops.to(device)
    model_for_flops.eval()

    dummy = torch.randn(*input_size).to(device)

    try:
        with torch.no_grad():
            flops, params = profile(model_for_flops, inputs=(dummy,), verbose=False)

        flops_info = {
            "flops": flops,
            "flops_g": flops / 1e9,
        }

    except Exception as e:
        print(f"[Warning] FLOPs measurement failed: {e}")
        flops_info = None

    # 清理副本
    del model_for_flops
    del dummy
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return flops_info

def clear_thop_hooks(model):
    """
    Remove possible residual hooks created by thop.
    This is a safety cleanup.
    """
    for m in model.modules():
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks.clear()
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
        if hasattr(m, "_backward_hooks"):
            m._backward_hooks.clear()

        if hasattr(m, "total_ops"):
            delattr(m, "total_ops")
        if hasattr(m, "total_params"):
            delattr(m, "total_params")


@torch.no_grad()
def measure_speed(model, data_loader, device, warmup_iters=10, speed_iters=50):
    """
    Measure throughput and latency.

    throughput: images / second
    latency_ms_per_image: milliseconds / image
    latency_ms_per_batch: milliseconds / batch
    """
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    total_images = 0
    total_time = 0.0
    measured_batches = 0

    data_iter = iter(data_loader)

    # warmup
    for _ in range(warmup_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        images = batch[0].to(device, non_blocking=True)

        _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # real timing
    for _ in range(speed_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)

        images = batch[0].to(device, non_blocking=True)
        batch_size = images.size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

        if device.type == "cuda":
            start.record()
            _ = model(images)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
        else:
            import time
            t0 = time.time()
            _ = model(images)
            elapsed = time.time() - t0

        total_time += elapsed
        total_images += batch_size
        measured_batches += 1

    throughput = total_images / max(total_time, 1e-12)
    latency_ms_per_image = 1000.0 * total_time / max(total_images, 1)
    latency_ms_per_batch = 1000.0 * total_time / max(measured_batches, 1)

    return {
        "throughput": throughput,
        "latency_ms_per_image": latency_ms_per_image,
        "latency_ms_per_batch": latency_ms_per_batch,
        "speed_batches": measured_batches,
        "speed_images": total_images,
    }

@torch.no_grad()
def measure_pure_inference_speed(model, data_loader, device, warmup_iters=50, speed_iters=200):
    """
    Measure pure model inference speed on GPU.

    This excludes image file reading, TIFF decoding, PIL transforms,
    and most DataLoader overhead by reusing one prepared batch.
    """
    model.eval()

    batch = next(iter(data_loader))
    images = batch[0].to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # warmup
    for _ in range(warmup_iters):
        _ = model(images)

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_images = images.size(0) * speed_iters

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(speed_iters):
            _ = model(images)
        end.record()

        torch.cuda.synchronize()
        total_time = start.elapsed_time(end) / 1000.0
    else:
        import time
        t0 = time.time()
        for _ in range(speed_iters):
            _ = model(images)
        total_time = time.time() - t0

    throughput = total_images / max(total_time, 1e-12)
    latency_ms_per_image = 1000.0 * total_time / max(total_images, 1)
    latency_ms_per_batch = 1000.0 * total_time / max(speed_iters, 1)

    return {
        "pure_throughput": throughput,
        "pure_latency_ms_per_image": latency_ms_per_image,
        "pure_latency_ms_per_batch": latency_ms_per_batch,
    }

@torch.no_grad()
def validate(model, data_loader, device, topk=(1, 5)):
    model.eval()

    correct = {k: 0 for k in topk}
    total = 0

    for batch in tqdm(data_loader, desc="Evaluating", ncols=100):
        images, labels = batch[0], batch[1]

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        logits = model(images)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)

        for k in topk:
            correct_k = pred[:, :k].eq(labels.view(-1, 1)).sum().item()
            correct[k] += correct_k

        total += labels.numel()

    results = {}
    for k in topk:
        results[f"top{k}"] = 100.0 * correct[k] / max(total, 1)

    results["total"] = total
    return results


def evaluate_one_variant(base_args, variant, device):
    print("=" * 100)
    print(f"Experiment    : {variant['name']}")
    print(f"Config        : {variant['cfg']}")
    print(f"Checkpoint    : {variant['resume']}")
    print(f"freq_override : {variant.get('freq_override', 'none')}")
    print(f"opts          : {variant.get('opts', [])}")
    print("=" * 100)

    config = make_config_for_variant(base_args, variant)

    model = build_model(config)
    print(f"Model built: {type(model).__name__}")
    
    param_info = count_params(model)
    print(f"Params: {param_info['params_m']:.4f} M")
    print(f"Trainable params: {param_info['trainable_params_m']:.4f} M")
    
    model = load_checkpoint(model, variant["resume"])
    model.to(device)
    model.eval()

    loader_outputs = build_loader(config)

    if base_args.eval_set == "val":
        eval_loader = loader_outputs[4]
        dataset_name = "Validation"
    else:
        eval_loader = loader_outputs[5]
        dataset_name = "Test"

    print(f"Using {dataset_name} set for evaluation.")
    print(f"{dataset_name} dataset size: {len(eval_loader.dataset)}")

    flops_info = None
    if base_args.measure_flops:
        flops_info = measure_flops(
            model,
            input_size=(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
            device=device
        )
    
        # 保险清理，避免 thop hook 污染后续测速和验证
        clear_thop_hooks(model)
    
        if flops_info is not None:
            print(f"FLOPs: {flops_info['flops_g']:.4f} G")
        else:
            print("FLOPs: None")
    
    speed_info = measure_speed(
        model,
        eval_loader,
        device,
        warmup_iters=base_args.warmup_iters,
        speed_iters=base_args.speed_iters
    )

    pure_speed_info = measure_pure_inference_speed(
        model,
        eval_loader,
        device,
        warmup_iters=50,
        speed_iters=200
    )
    
    print(f"Pure throughput: {pure_speed_info['pure_throughput']:.2f} images/s")
    print(f"Pure latency: {pure_speed_info['pure_latency_ms_per_image']:.4f} ms/image")
    print(f"Pure batch latency: {pure_speed_info['pure_latency_ms_per_batch']:.4f} ms/batch")
        
    print(f"Throughput: {speed_info['throughput']:.2f} images/s")
    print(f"Latency: {speed_info['latency_ms_per_image']:.4f} ms/image")
    print(f"Batch latency: {speed_info['latency_ms_per_batch']:.4f} ms/batch")

    results = validate(model, eval_loader, device, topk=(1, 5))

    print(f"{variant['name']} {dataset_name} results:")
    print(f"  top1 : {results['top1']:.4f}%")
    print(f"  top5 : {results['top5']:.4f}%")
    print(f"  total: {results['total']}")

    return {
        "name": variant["name"],
        "eval_set": base_args.eval_set,
    
        "top1": results["top1"],
        "top5": results["top5"],
        "total": results["total"],
    
        "params_m": param_info["params_m"],
        "trainable_params_m": param_info["trainable_params_m"],
    
        "flops_g": flops_info["flops_g"] if flops_info is not None else None,
    
        "throughput": speed_info["throughput"],
        "latency_ms_per_image": speed_info["latency_ms_per_image"],
        "latency_ms_per_batch": speed_info["latency_ms_per_batch"],
    
        "cfg": variant["cfg"],
        "resume": variant["resume"],
        "freq_override": variant.get("freq_override", "none"),

        "pure_throughput": pure_speed_info["pure_throughput"],
        "pure_latency_ms_per_image": pure_speed_info["pure_latency_ms_per_image"],
        "pure_latency_ms_per_batch": pure_speed_info["pure_latency_ms_per_batch"],
    }


def save_csv(results, csv_path):
    import csv

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    fieldnames = [
        "name",
        "eval_set",
        "top1",
        "top5",
        "total",
    
        "params_m",
        "trainable_params_m",
        "flops_g",
        "throughput",
        "latency_ms_per_image",
        "latency_ms_per_batch",
    
        "cfg",
        "resume",
        "freq_override",

        "pure_throughput",
        "pure_latency_ms_per_image",
        "pure_latency_ms_per_batch",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Saved CSV summary to: {csv_path}")


def print_summary(results):
    print("\n" + "=" * 140)
    print("Final summary")
    print("=" * 140)

    print(
        f"{'Experiment':<16} "
        f"{'EvalSet':<8} "
        f"{'Top-1 (%)':>10} "
        f"{'Top-5 (%)':>10} "
        f"{'Params(M)':>10} "
        f"{'FLOPs(G)':>10} "
        f"{'Throughput':>14} "
        f"{'ms/img':>10} "
        f"{'Total':>8}"
    )

    print("-" * 140)

    for r in results:
        flops_str = f"{r['flops_g']:.4f}" if r["flops_g"] is not None else "N/A"

        print(
            f"{r['name']:<16} "
            f"{r['eval_set']:<8} "
            f"{r['top1']:>10.4f} "
            f"{r['top5']:>10.4f} "
            f"{r['params_m']:>10.4f} "
            f"{flops_str:>10} "
            f"{r['throughput']:>14.2f} "
            f"{r['latency_ms_per_image']:>10.4f} "
            f"{r['total']:>8}"
        )

    print("=" * 140)


def main():
    args = parse_option()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    variants = load_variants(args.variants_json)

    print(f"Loaded {len(variants)} variants from:")
    print(args.variants_json)

    all_results = []

    for variant in variants:
        result = evaluate_one_variant(args, variant, device)
        all_results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary(all_results)

    if args.output_csv is not None:
        save_csv(all_results, args.output_csv)


if __name__ == "__main__":
    main()
