import glob
import numpy as np

from inference.predict import predict
from utils.torch import load_model_weights
from utils.metrics import study_level_map
from sklearn.metrics import roc_auc_score

from data.dataset import CovidInfDataset
from data.transforms import get_tranfos_inference
from model_zoo.models import get_model

from params import CLASSES


def inference_val(
    config,
    log_folder,
    df,
    root_dir="",
    flip_tta=False,
):
    """
    Inference on the validation data.
    Args:
        config (Config): Parameters.
        log_folder (string): Path to experiment.
        df (pandas dataframe): Metadata.
        root_dir (str): Directory with all the images. Defaults to "".
        flip_tta (bool, optional): Whether to apply horizontal flip tta. Defaults to False.
    """

    model = get_model(
        config.selected_model,
        num_classes=config.num_classes,
    ).to(config.device)
    model.zero_grad()

    weights = sorted(glob.glob(log_folder + "*.pt"))
    assert len(weights) == config.k

    pred_oof_study = np.zeros((len(df), config.num_classes))
    pred_oof_img = np.zeros(len(df))
    for i, weight in enumerate(weights):
        load_model_weights(model, weight)

        df_val = df[df[config.folds_col] == i].copy().reset_index(drop=True)
        dataset = CovidInfDataset(
            df_val,
            root_dir=root_dir,
            transforms=get_tranfos_inference(mean=model.mean, std=model.std),
        )

        pred_study, pred_img = predict(
            model,
            dataset,
            batch_size=config.val_bs,
            num_classes=config.num_classes,
            flip_tta=flip_tta,
            device=config.device,
        )

        val_idx = np.array(df[df[config.folds_col] == i].index)
        pred_oof_study[val_idx] = pred_study
        pred_oof_img[val_idx] = pred_img

    if log_folder is not None:
        suffix = "_flip" * flip_tta
        np.save(log_folder + 'pred_oof_study' + suffix + '.npy', pred_oof_study)
        np.save(log_folder + 'pred_oof_img' + suffix + '.npy', pred_oof_img)

    score_study = study_level_map(
        pred_oof_study, df[CLASSES].values, df["study_id"]
    )
    score_img = roc_auc_score(df["img_target"].values, pred_oof_img)

    print('CV Scores :')
    print(f' -> Study mAP : {score_study :.4f}')
    print(f' -> Image AUC : {score_img :.4f}')

    return pred_oof_study, pred_oof_img
