# ------------------------------------------------------------
# Model builder for vHeat with PHR/FDR.
#
# Supported variants:
#   1. Original vHeat
#   2. PHR-only
#   3. FDR-only
#   4. PHR + FDR
#
# Note:
#   PARTIAL_HEAT corresponds to Partial Heat Routing (PHR) in the paper.
#   FP corresponds to Frequency-domain Regularization (FDR) in the paper.
# ------------------------------------------------------------

from .vHeat import vHeat


def build_vHeat_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # ============================================================
    # Resolve Partial Heat Routing (PHR) config
    # ============================================================
    partial_heat_enable = config.MODEL.PARTIAL_HEAT.ENABLE
    partial_heat_mode = config.MODEL.PARTIAL_HEAT.MODE

    block_heat_ratios = None

    if partial_heat_mode == "stage":
        stage_heat_ratios = list(config.MODEL.PARTIAL_HEAT.STAGE_HEAT_RATIOS)
        heat_ratio = config.MODEL.PARTIAL_HEAT.HEAT_RATIO

    elif partial_heat_mode == "global":
        stage_heat_ratios = None
        heat_ratio = config.MODEL.PARTIAL_HEAT.HEAT_RATIO

    elif partial_heat_mode == "block":
        stage_heat_ratios = list(config.MODEL.PARTIAL_HEAT.STAGE_HEAT_RATIOS)
        heat_ratio = config.MODEL.PARTIAL_HEAT.HEAT_RATIO

        if hasattr(config.MODEL.PARTIAL_HEAT, "BLOCK_HEAT_RATIOS"):
            block_heat_ratios = {
                str(k): float(v)
                for k, v in config.MODEL.PARTIAL_HEAT.BLOCK_HEAT_RATIOS.items()
            }
        else:
            block_heat_ratios = {}

    else:
        raise ValueError(f"Unsupported PARTIAL_HEAT mode: {partial_heat_mode}")

    light_branch_type = config.MODEL.PARTIAL_HEAT.LIGHT_BRANCH

    # ============================================================
    # Resolve Frequency-domain Regularization (FDR) config
    #
    # The original config name FP is retained for compatibility.
    # It controls stage-wise frequency retention ratios used by FDR.
    # ============================================================
    fp_enable = config.MODEL.FP.ENABLE
    fp_mode = config.MODEL.FP.MODE

    if fp_mode == "stage":
        stage_freq_ratios = list(config.MODEL.FP.STAGE_RATIOS)
        freq_keep_ratio = config.MODEL.FP.KEEP_RATIO

    elif fp_mode == "global":
        stage_freq_ratios = None
        freq_keep_ratio = config.MODEL.FP.KEEP_RATIO

    else:
        raise ValueError(f"Unsupported FP mode: {fp_mode}")

    # ============================================================
    # Build vHeat
    # ============================================================
    if model_type == "vHeat":
        model = vHeat(
            in_chans=config.MODEL.VHEAT.IN_CHANS,
            patch_size=config.MODEL.VHEAT.PATCH_SIZE,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VHEAT.DEPTHS,
            dims=config.MODEL.VHEAT.DIMS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_ratio=config.MODEL.VHEAT.MLP_RATIO,
            post_norm=config.MODEL.VHEAT.POST_NORM,
            layer_scale=config.MODEL.VHEAT.LAYER_SCALE,
            img_size=config.DATA.IMG_SIZE,
            infer_mode=config.EVAL_MODE or config.THROUGHPUT_MODE,

            # PHR.
            partial_heat=partial_heat_enable,
            heat_ratio=heat_ratio,
            stage_heat_ratios=stage_heat_ratios,
            block_heat_ratios=block_heat_ratios,
            light_branch_type=light_branch_type,

            # FDR.
            freq_pruning=fp_enable,
            freq_keep_ratio=freq_keep_ratio,
            stage_freq_ratios=stage_freq_ratios,
        )

        if config.THROUGHPUT_MODE:
            model.infer_init()

        return model

    raise ValueError(f"Unsupported model type: {model_type}")


def build_model(config, is_pretrain=False):
    return build_vHeat_model(config, is_pretrain)
