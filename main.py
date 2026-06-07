import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
from timm.utils import ModelEma

from fvcore.nn import FlopCountAnalysis, flop_count_str

from utils.config import get_config
from models import build_model
from data import build_loader
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import (
    save_checkpoint,
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    reduce_tensor,
)
from utils.utils_ema import (
    load_checkpoint_ema,
    load_pretrained_ema,
    save_checkpoint_ema,
    save_ema_best_checkpoint,
)


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True

    if v.lower() in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_option():
    parser = argparse.ArgumentParser(
        "vHeat pollen recognition training and evaluation script",
        add_help=False,
    )

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options by adding KEY VALUE pairs.",
        default=None,
        nargs="+",
    )

    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--pretrained", help="pretrained weight from checkpoint")
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")

    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )

    parser.add_argument(
        "--disable_amp",
        action="store_true",
        help="Disable PyTorch AMP",
    )

    parser.add_argument(
        "--amp-opt-level",
        type=str,
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, deprecated",
    )

    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder",
    )

    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")

    parser.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="local rank for DistributedDataParallel",
    )

    parser.add_argument(
        "--fused_layernorm",
        action="store_true",
        help="Use fused layernorm.",
    )

    parser.add_argument(
        "--optim",
        type=str,
        help="overwrite optimizer if provided, e.g. adamw/sgd/fused_adam/fused_lamb",
    )

    parser.add_argument("--model_ema", type=str2bool, default=True)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999)
    parser.add_argument("--model_ema_force_cpu", type=str2bool, default=False)

    args, _ = parser.parse_known_args()

    if args.local_rank is None:
        args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    config = get_config(args)

    return args, config


def main(config, args):
    torch.cuda.empty_cache()

    (
        dataset_train,
        dataset_val,
        dataset_test,
        data_loader_train,
        data_loader_val,
        data_loader_test,
        mixup_fn,
    ) = build_loader(config)

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    try:
        input_tensor = dataset_val[0][0][None]
        logger.info(flop_count_str(FlopCountAnalysis(model, (input_tensor,))))
    except Exception as e:
        logger.info(str(model))
        logger.info(f"FLOP analysis failed: {e}")
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model, "flops"):
            flops = model.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        logger.info(f"Using EMA with decay = {args.model_ema_decay:.8f}")

    optimizer = build_optimizer(config, model)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(
            config,
            optimizer,
            len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS,
        )
    else:
        lr_scheduler = build_scheduler(
            config,
            optimizer,
            len(data_loader_train),
        )

    if config.AUG.MIXUP > 0.0:
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0
    max_accuracy_ema = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)

        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from "
                    f"{config.MODEL.RESUME} to {resume_file}"
                )

            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()

            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        model_without_ddp, max_accuracy, max_accuracy_ema = load_checkpoint_ema(
            config,
            model_without_ddp,
            optimizer,
            lr_scheduler,
            loss_scaler,
            logger,
            model_ema,
        )

        if config.DATA.VAL_HAS_LABELS:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(
                f"Accuracy of the network on the {len(dataset_val)} validation images: "
                f"{acc1:.1f}%"
            )

            if model_ema is not None:
                acc1_ema, acc5_ema, loss_ema = validate(
                    config,
                    data_loader_val,
                    model_ema.ema,
                )
                logger.info(
                    f"Accuracy of the EMA network on the {len(dataset_val)} validation images: "
                    f"{acc1_ema:.1f}%"
                )
        else:
            logger.info("Validation set has no labels, skip accuracy calculation.")

        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and not config.MODEL.RESUME:
        load_pretrained_ema(
            config,
            model_without_ddp,
            logger,
            model_ema,
        )

        if config.DATA.VAL_HAS_LABELS:
            acc1, acc5, loss = validate(config, data_loader_val, model)
            logger.info(
                f"Accuracy of the network on the {len(dataset_val)} validation images: "
                f"{acc1:.1f}%"
            )

            if model_ema is not None:
                acc1_ema, acc5_ema, loss_ema = validate(
                    config,
                    data_loader_val,
                    model_ema.ema,
                )
                logger.info(
                    f"Accuracy of the EMA network on the {len(dataset_val)} validation images: "
                    f"{acc1_ema:.1f}%"
                )
        else:
            logger.info("Validation set has no labels, skip pretrained model accuracy check.")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)

        if model_ema is not None:
            throughput(data_loader_val, model_ema.ema, logger)

        return

    logger.info("Start training")

    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if hasattr(data_loader_train.sampler, "set_epoch"):
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config=config,
            model=model,
            criterion=criterion,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            mixup_fn=mixup_fn,
            lr_scheduler=lr_scheduler,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
        )

        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 0
            or epoch == config.TRAIN.EPOCHS - 1
        ):
            save_checkpoint_ema(
                config,
                epoch,
                model_without_ddp,
                max_accuracy,
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
                model_ema,
                max_accuracy_ema,
            )

        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(
            f"Accuracy of the network on the {len(dataset_val)} validation images: "
            f"{acc1:.1f}%"
        )

        if dist.get_rank() == 0 and acc1 > max_accuracy:
            save_checkpoint(
                config,
                epoch,
                model,
                acc1,
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
                best="best",
            )

        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")

        if model_ema is not None:
            acc1_ema, acc5_ema, loss_ema = validate(
                config,
                data_loader_val,
                model_ema.ema,
            )
            logger.info(
                f"Accuracy of the EMA network on the {len(dataset_val)} validation images: "
                f"{acc1_ema:.1f}%"
            )

            if dist.get_rank() == 0 and acc1_ema > max_accuracy_ema:
                save_ema_best_checkpoint(
                    config,
                    epoch,
                    model_without_ddp,
                    model_ema,
                    acc1_ema,
                    optimizer,
                    lr_scheduler,
                    loss_scaler,
                    logger,
                    max_accuracy=max_accuracy,
                )

            max_accuracy_ema = max(max_accuracy_ema, acc1_ema)
            logger.info(f"Max accuracy ema: {max_accuracy_ema:.2f}%")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    loss_scaler,
    model_ema=None,
):
    model.train()
    optimizer.zero_grad()
    torch.cuda.reset_peak_memory_stats()

    num_steps = len(data_loader)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            logits = model(samples)
            total_loss = criterion(logits, targets)
            cls_loss = total_loss

        loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        is_second_order = (
            hasattr(optimizer, "is_second_order")
            and optimizer.is_second_order
        )

        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(),
            create_graph=is_second_order,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )

            if model_ema is not None:
                model_ema.update(model)

        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(total_loss.item(), targets.size(0))
        cls_loss_meter.update(cls_loss.item(), targets.size(0))

        if grad_norm is not None:
            norm_meter.update(grad_norm)

        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            wd = optimizer.param_groups[0]["weight_decay"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)

            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))}\t"
                f"lr {lr:.6f}\t"
                f"wd {wd:.4f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"data time {data_time.val:.4f} ({data_time.avg:.4f})\t"
                f"total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"cls_loss {cls_loss_meter.val:.4f} ({cls_loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes "
        f"{datetime.timedelta(seconds=int(epoch_time))}"
    )

    if (epoch + 1) % 10 == 0 and config.LOCAL_RANK == 0:
        checkpoint_dir = config.OUTPUT
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
                if not hasattr(model, "module")
                else model.module.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "loss_scaler_state_dict": loss_scaler.state_dict(),
            "total_loss_avg": loss_meter.avg,
            "cls_loss_avg": cls_loss_meter.avg,
        }

        if model_ema is not None:
            checkpoint["model_ema_state_dict"] = model_ema.ema.state_dict()

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"ckpt_epoch_{epoch + 1}.pth",
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to: {checkpoint_path}")


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).long()

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t"
                f"Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t"
                f"Mem {memory_used:.0f}MB"
            )

    logger.info(
        f" * Acc@1 {acc1_meter.avg:.3f} "
        f"Acc@5 {acc5_meter.avg:.3f}"
    )

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]

        for _ in range(50):
            model(images)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        logger.info("throughput averaged with 30 times")

        tic1 = time.time()

        for _ in range(30):
            model(images)

        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        torch.cuda.synchronize()
        tic2 = time.time()

        throughput_value = 30 * batch_size / (tic2 - tic1)

        logger.info(f"batch_size {batch_size} throughput {throughput_value}")
        logger.info(f"Mem {memory_used:.0f}MB")

        return


if __name__ == "__main__":
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex AMP has been deprecated, please use PyTorch AMP instead.")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(config.LOCAL_RANK)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    linear_scaled_lr = (
        config.TRAIN.BASE_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR
        * config.DATA.BATCH_SIZE
        * dist.get_world_size()
        / 512.0
    )

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr *= config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr *= config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)

    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=dist.get_rank(),
        name=f"{config.MODEL.NAME}",
    )

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())

        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, args)
