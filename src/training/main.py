import gc
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from params import CLASSES
from training.train import fit
from data.dataset import CovidClsDataset
from data.transforms import get_transfos_cls
from model_zoo.models import get_model
from utils.metrics import study_level_map
from utils.torch import seed_everything, count_parameters, save_model_weights


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
        np array [n x num_classes]: Study validation predictions.
        np array [n x 1]: Image validation predictions.
    """

    seed_everything(config.seed)

    model = get_model(
        config.selected_model,
        num_classes=config.num_classes,
    ).to(config.device)
    model.zero_grad()

    train_dataset = CovidClsDataset(
        df_train,
        transforms=get_transfos_cls(augment=True, mean=model.mean, std=model.std),
    )

    val_dataset = CovidClsDataset(
        df_val,
        transforms=get_transfos_cls(augment=False, mean=model.mean, std=model.std),
    )

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters")

    pred_val_study, pred_val_img = fit(
        model,
        train_dataset,
        val_dataset,
        config.loss_config,
        samples_per_patient=config.samples_per_patient,
        optimizer=config.optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        mix=config.mix,
        mix_proba=config.mix_proba,
        mix_alpha=config.mix_alpha,
        num_classes=config.num_classes,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        use_fp16=config.use_fp16,
        device=config.device,
    )

    if config.save_weights and log_folder is not None:
        save_model_weights(
            model,
            f"{config.selected_model}_{config.name}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, train_dataset, val_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    return pred_val_study, pred_val_img


def k_fold(config, df, df_extra=None, log_folder=None):
    """
    Performs a patient grouped k-fold cross validation.

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        df_extra (None pandas dataframe, optional): Additional training data. Defaults to None.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        np array [N x num_classes]: Study oof predictions.
        np array [N x 1]: Image oof predictions.
    """

    pred_oof_study = np.zeros((len(df), config.num_classes))
    pred_oof_img = np.zeros(len(df))

    for i in range(config.k):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df[df[config.folds_col] != i].copy().reset_index(drop=True)
            df_val = df[df[config.folds_col] == i].copy().reset_index(drop=True)

            if df_extra is not None:
                df_train = pd.concat([df_train, df_extra], sort=False).reset_index(
                    drop=True
                )

            for study in df_train['study_id'].unique():
                assert study not in df_val['study_id'].values

            pred_val_study, pred_val_img = train(
                config, df_train, df_val, i, log_folder=log_folder
            )

            val_idx = np.array(df[df[config.folds_col] == i].index)
            pred_oof_study[val_idx] = pred_val_study
            pred_oof_img[val_idx] = pred_val_img

            if log_folder is not None:
                np.save(log_folder + f"pred_val_study_{i}.npy", pred_val_study)
                np.save(log_folder + f"pred_val_img_{i}.npy", pred_val_img)
            else:
                return pred_val_study, pred_val_img

    if log_folder is not None:
        np.save(log_folder + "pred_oof_study.npy", pred_oof_study)
        np.save(log_folder + "pred_oof_img.npy", pred_oof_img)

    score_study = study_level_map(
        pred_oof_study, df[CLASSES].values, df["study_id"]
    )

    score_img = roc_auc_score(df["img_target"].values, pred_oof_img)

    print('CV Scores :')
    print(f' -> Study mAP : {score_study :.4f}')
    print(f' -> Image AUC : {score_img :.4f}')

    return pred_oof_study, pred_oof_img
