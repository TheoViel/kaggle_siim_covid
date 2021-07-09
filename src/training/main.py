import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from training.train import fit
from data.dataset import ColorBCCDataset
from data.transforms import OCT_preprocess
from model_zoo.models import define_model
from utils.metrics import compute_metrics, SeededGroupKFold
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
        np array: Validation predictions.
        pandas dataframe: Training history.
    """

    seed_everything(config.seed)

    model = define_model(
        config.selected_model,
        use_mixstyle=config.use_mixstyle,
        use_attention=config.use_attention,
        reduce_stride_3=config.reduce_stride_3,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        pretrained_weights=config.pretrained_weights,
    ).to(config.device)
    model.zero_grad()

    if config.mean_teacher_config is not None:
        mean_teacher = define_model(
            config.selected_model,
            use_mixstyle=config.use_mixstyle,
            use_attention=config.use_attention,
            reduce_stride_3=config.reduce_stride_3,
            num_classes=config.num_classes,
            num_classes_aux=config.num_classes_aux,
            pretrained_weights=config.pretrained_weights,
        ).to(config.device)
        mean_teacher.zero_grad()
        for param in mean_teacher.parameters():
            param.detach_()
    else:
        mean_teacher = None

    train_dataset = ColorBCCDataset(
        df_train,
        root_dir=config.img_folder,
        img_name=config.img_name,
        transforms=OCT_preprocess(),
        target_name=config.target_name,
        train=True,
    )

    val_dataset = ColorBCCDataset(
        df_val,
        root_dir=config.img_folder,
        img_name=config.img_name,
        transforms=OCT_preprocess(augment=False),
        target_name=config.target_name,
    )

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val, history = fit(
        model,
        mean_teacher,
        config.mean_teacher_config,
        train_dataset,
        val_dataset,
        samples_per_patient=config.samples_per_patient,
        optimizer_name=config.optimizer,
        loss_name=config.loss,
        activation=config.activation,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        swa_first_epoch=config.swa_first_epoch,
        mix=config.mix,
        mix_proba=config.mix_proba,
        mix_alpha=config.mix_alpha,
        num_classes=config.num_classes,
        aux_loss_weight=config.aux_loss_weight,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
    )

    if config.save_weights and log_folder is not None:
        save_model_weights(
            model,
            f"{config.selected_model}_{config.name}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, mean_teacher, train_dataset, val_dataset)
    torch.cuda.empty_cache()

    return pred_val, history


def get_metrics_cv(df, pred_oof, config):
    """
    Computes the metrics for the out of fold predictions.

    Args:
        df (pandas dataframe): Metadata, contains the targets.
        pred_oof (np array [N x NUM_CLASSES]): Out of fold predictions.
        config (Config): Config.

    Returns:
        dict : Metrics dictionary.
    """
    if config.num_classes > 1:
        pred_cols = []
        for i in range(pred_oof.shape[1]):
            df[f'pred_{i}'] = pred_oof[:, i]
            pred_cols.append(f'pred_{i}')

        df = df[df[config.target_name_eval] >= 0]  # class -1 is not used

        metrics = compute_metrics(
            df[pred_cols].values,
            df[config.target_name_eval].values,
            num_classes=config.num_classes,
            loss_name=config.loss
        )

        print(f"\n  -> CV accuracy : {metrics['accuracy'][0]:.3f}")
    else:
        df['pred'] = pred_oof

        df = df[df[config.target_name_eval] >= 0]  # class -1 is not used
        metrics = compute_metrics(
            df['pred'].values,
            df[config.target_name_eval].values,
            num_classes=config.num_classes,
            loss_name=config.loss
        )

        print(f"\n  -> CV auc : {metrics['auc'][0]:.3f}")

    return metrics


def k_fold(config, df, df_extra=None, log_folder=None):
    """
    Performs a patient grouped k-fold cross validation.
    The extra data is used for training at each fold.
    The following things are saved to the log folder :
    oof predictions, val predictions, val indices, histories.

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        df_extra (None pandas dataframe, optional): Additional training data. Defaults to None.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        dict : Metrics dictionary.
    """
    if config.class_weights is not None:
        df["sample_weight"] = df["target"].apply(lambda x: config.class_weights[int(x)])

    if config.random_state == 0:
        gkf = GroupKFold(n_splits=config.k)
    else:
        gkf = SeededGroupKFold(n_splits=config.k, random_state=config.random_state)

    splits = list(gkf.split(X=df, y=df, groups=df["split"]))
    pred_oof = np.zeros((len(df), config.num_classes))

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            # print(df_val[['target', 'split']].groupby('split').agg(lambda x: np.unique(list(x))))
            # df_train = df_train[df_train['target_unverif'] != "UNK"]

            if df_extra is not None:
                val_split = df_val["split"].unique()
                df_extra_ = df_extra[~df_extra["split"].isin(val_split)]

                df_train = pd.concat([df_train, df_extra_], sort=False).reset_index(
                    drop=True
                )

            for patient in df_train['patient'].unique():
                assert patient not in df_val['patient'].values

            pred_val, history = train(
                config, df_train, df_val, i, log_folder=log_folder
            )
            pred_oof[val_idx] = pred_val

            if log_folder is not None:
                np.save(log_folder + f"val_idx_{i}.npy", val_idx)
                np.save(log_folder + f"pred_val_{i}.npy", pred_val)
                history.to_csv(log_folder + f"history_{i}.csv", index=False)
            else:
                return

    if log_folder is not None:
        np.save(log_folder + "pred_oof.npy", pred_oof)

    metrics = get_metrics_cv(df, pred_oof, config)
    if log_folder is not None:
        metrics = pd.DataFrame.from_dict(metrics)
        metrics.to_csv(log_folder + "metrics.csv", index=False)

    return metrics
