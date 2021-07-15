import torch
from sklearn.model_selection import GroupKFold

from model import define_model
from training.train import fit
from data.dataset import FollicleDataset
from data.transforms import OCT_preprocess
from util.torch_utils import seed_everything, count_parameters, save_model_weights


def train(config, df_train, df_val, fold, log_folder=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        DetectionMeter: Meter.
    """
    seed_everything(config.seed)

    model = define_model(config).to(config.device)

    train_dataset = FollicleDataset(
        df_train,
        transforms=OCT_preprocess(augment=True, bbox_format=config.bbox_format),
        mosaic_proba=config.mosaic_proba,
        root=config.img_folder,
        bbox_format=config.bbox_format,
    )

    val_dataset = FollicleDataset(
        df_val,
        transforms=OCT_preprocess(augment=False, bbox_format=config.bbox_format),
        root=config.img_folder,
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
            f"{config.selected_model}_{fold}.pt",
            cp_folder=log_folder,
        )

    del model, train_dataset, val_dataset
    torch.cuda.empty_cache()

    return meter


def k_fold_training(config, df, log_folder=None):
    """
    Performs a patient grouped k-fold cross validation.
    The following things are saved to the log folder : val predictions, val indices, histories

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        list of DetectionMeter: Meters.
    """

    gkf = GroupKFold(n_splits=config.k)

    df["groups"] = df["patient"]
    splits = list(gkf.split(X=df, y=df, groups=df["groups"]))

    meters = []
    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()

            for patient in df_train["groups"].unique():
                assert patient not in df_val["groups"].values

            meter = train(config, df_train, df_val, i, log_folder=log_folder)

            if log_folder is not None:
                # save stuff
                pass

            if log_folder is None or len(config.selected_folds) == 1:
                meter.val_idx = val_idx
                meters.append(meter)
                return meters

    return meters
