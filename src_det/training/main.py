import torch

from model import define_model
from training.train import fit
from data.dataset import LungDataset
from data.transforms import get_transfos_lung
from util.torch_utils import seed_everything, count_parameters, save_model_weights


def train(config, boxes_dic_cxr, boxes_dic_siim, log_folder=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        DetectionMeter: Meter.
    """
    seed_everything(config.seed)

    model = define_model(config).to(config.device)

    keys = list(boxes_dic_siim.keys())
    boxes_dic_siim_train = {k: boxes_dic_siim[k] for k in keys[25:]}
    boxes_dic_siim_val = {k: boxes_dic_siim[k] for k in keys[:25]}

    train_dataset = LungDataset(
        boxes_dic_cxr,
        boxes_dic_siim_train,
        transforms=get_transfos_lung(
            augment=True, mean=model.mean, std=model.std, bbox_format=config.bbox_format
        ),
        bbox_format=config.bbox_format,
    )

    val_dataset = LungDataset(
        {},
        boxes_dic_siim_val,
        transforms=get_transfos_lung(
            augment=False, mean=model.mean, std=model.std, bbox_format=config.bbox_format
        ),
        bbox_format=config.bbox_format,
    )

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    meter = fit(
        model,
        config,
        train_dataset,
        val_dataset,
    )

    if config.save_weights and log_folder is not None:
        save_model_weights(
            model,
            f"{config.selected_model}.pt",
            cp_folder=log_folder,
        )

    del model, train_dataset, val_dataset
    torch.cuda.empty_cache()

    return meter
