import re
import numpy as np
from params import CLASSES

ANOMALIES = [
    "IM07_3D_P0310_L02_s_and_nBCC_study3D_L02_RI_FA_23-10-2019",
    "IM01_3D_P0310_L02_s_and_nBCC_study3D_L01_RI_FA_23-10-2019",
    "IM08_3D_P0310_L02_s_and_nBCC_study3D_L01_RI_FA_23-10-2019",
    "IM02_3D_P0310_L02_s_and_nBCC_study3D_L01_RI_FA_23-10-2019",
    "IM11_3D_P0051_L02_and_L03_Bowen_L05_and_L09_SCC_in_situ_study3D_L02_ZE_VA_10-12-2019",
    "IM03_3D_P0304_L01_nBCC_L08_residu_keratoacanthoma_study3D_L07_CH_HE_22-10-2019",
    "IM05_3D_P0304_L01_nBCC_L08_residu_keratoacanthoma_study3D_L07_CH_HE_22-10-2019",
    "IM02_3D_P0304_L01_nBCC_L08_residu_keratoacanthoma_study3D_L06_CH_HE_22-10-2019",
    "IM01_3D_P0304_L01_nBCC_L08_residu_keratoacanthoma_study3D_L06_CH_HE_22-10-2019",
    "IM16_3D_P0376_nBCC_study3D_L01_GH_MI_10-12-2019",
    "IM03_3D_P0514_L01_SCC_in_situ_L02_verrue_study3D_L01_WI_VA_04-05-2020",
    "IM05_3D_P0337_AK_study3D_L04_DO_WI_07-01-2020",
    "IM09_3D_P0337_AK_study3D_L04_DO_WI_07-01-2020",
    "IM07_3D_P0337_AK_study3D_L04_DO_WI_07-01-2020",
    "IM07_3D_P0036_AK_study3D_L01_CH_ME_28-05-2019",
    "IM12_3D_P0036_AK_study3D_L01_CH_ME_28-05-2019",
    "IM07_3D_P0321_L01_Melanoma_L0204_sBCC_L03_Angioleiomyofibrome_study3D_L01_CO_GE_28-10-2019",
    "IM10_3D_P0361_L0105_s_and_nBCC_L04_sBCC_L06_s_and_iBCC_study3D_L05_PA_TR_26-11-2019",
    "IM01_3D_P0362_L01_s_and_iBCC_study3D_L08_GI_DO_26-11-2019",
    "IM05_3D_P0119_L0304_SCC_study3D_L01_VE_LE_17-07-2019",
    "IM17_3D_P0414_L03_SCC_invasive_L05_AK_L06_SCC_in_situ_study3D_L04_MA_LU_14-01-2020",
    "IM04_3D_P0109_L01_keratoacanthome__SCC_invasive_study3D_L01_NI_JA_16-07-2019",
]


def prepare_test_data(df):
    """
    Prepares the dataframe for the bcc test data.

    Args:
        df (pandas dataframe): Test dataframe.

    Returns:
        pandas dataframe: Prepared dataframe.
    """
    # Remove outliers
    df = df[df.original_size.apply(lambda x: len(x.split(","))) == 2].reset_index(
        drop=True
    )
    df = df[~df.img_name.apply(lambda x: "THUMB" in x)].reset_index(drop=True)
    # df = df[~df.img_name.apply(lambda x: "secondary" in x)].reset_index(drop=True)

    # Img_id, frame, date and nb_frame columns
    df["patient"] = df["img_name"].apply(lambda x: "_".join(x.split("_")[2:8]))
    df["site"] = df["patient"].apply(lambda x: x.split("_")[-2])

    df["img_id"] = df["img_name"].apply(lambda x: "_".join(x.split("_")[1:-1]))
    df["frame"] = df["img_name"].apply(lambda x: int(x[:-4].split("_")[-1]))
    df["date"] = df["img_name"].apply(
        lambda x: re.search(r"(\d{1,2}-\d{1,2}-\d{1,4})", x).group(0)
    )

    nb_frames = (
        df.groupby(["img_id"])[["frame"]]
        .apply(max)
        .rename(columns={"frame": "nb_frames"})
    )
    df = df.merge(nb_frames, on=["img_id"])

    df["sample_weight"] = 1

    return df


def prepare_new_data(df):
    """
    Prepares the dataframe for the new bcc data.

    Args:
        df (pandas dataframe): Test dataframe.

    Returns:
        pandas dataframe: Prepared dataframe.
    """
    # Img_id, frame, date and nb_frame columns
    df["site"] = df["patient_id"].apply(lambda x: "ERASME" + x.split("_")[0])

    df["img_id"] = df["img_name"].apply(lambda x: "_".join(x.split("_")[:-2]))
    df["frame"] = df["img_name"].apply(lambda x: int(x[:-4].split("_")[-1]))

    # Anomalies
    df.loc[df['is_anomaly'], 'is_bcc'] = 0
    df.loc[df['img_id'].apply(lambda x: x in ANOMALIES), 'is_bcc'] = 0

    nb_frames = (
        df.groupby(["img_id"])[["frame"]]
        .agg(lambda x: len(list(x)))
        .rename(columns={"frame": "nb_frames"})
    )
    df = df.merge(nb_frames, on=["img_id"])

    df["target"] = df["target"].apply(lambda x: CLASSES.index(x))
    df["is_bcc_train"] = np.where(
        df["is_bcc"] == -1, (df["target_unverif"] == "BCC").astype(int), df["is_bcc"]
    )

    df["patient"] = df["patient_id"]
    df["split"] = df["site"] + "__" + df["patient"]

    df["sample_weight"] = 1  # np.where(df["is_bcc"] != -1, 1, 0.1)

    return df


def prepare_data_pretraining(df):
    """
    Cleaning for the pretraining data.

    Args:
        df (pandas dataframe): Train dataframe.

    Returns:
        pandas dataframe: Cleaned data.
    """
    # Remove outliers
    df = df[df.original_size.apply(lambda x: len(x.split(','))) == 2].reset_index(drop=True)
    df = df[~df.img_name.apply(lambda x: "THUMB" in x)].reset_index(drop=True)
    df = df[df['target_name'] != 'unused']

    # Img_id, frame, date and nb_frame columns
    df['img_id'] = df['img_name'].apply(lambda x: "_".join(x.split('_')[1:-1]))
    df['frame'] = df['img_name'].apply(lambda x: int(x[:-4].split('_')[-1]))
    df['date'] = df['img_name'].apply(lambda x: re.search(r'(\d{1,2}-\d{1,2}-\d{1,4})', x).group(0))

    nb_frames = df.groupby(['img_id'])[['frame']].apply(max).rename(columns={'frame': 'nb_frames'})
    df = df.merge(nb_frames, on=['img_id'])

    df['sample_weight'] = 1

    return df


def prepare_extra_data(df):
    """
    Prepares the dataframe for the extra data.

    Args:
        df (pandas dataframe): Test dataframe.

    Returns:
        pandas dataframe: Prepared dataframe.
    """
    # keep only 2Ds & videos

    # Img_id, frame, date and nb_frame columns
    df["site"] = ""

    df["img_id"] = df["img_name"].apply(lambda x: x.rsplit('_', 1)[0])
    df["frame"] = df["img_name"].apply(lambda x: int(x.rsplit('_', 1)[1]))

    df["img_name"] += ".png"

    nb_frames = (
        df.groupby(["img_id"])[["frame"]]
        .agg(lambda x: len(list(x)))
        .rename(columns={"frame": "nb_frames"})
    )
    df = df.merge(nb_frames, on=["img_id"])

    df["target"] = df["target"].apply(lambda x: "UNK" if x == "HEALTHY" else x)
    df["target"] = df["target"].apply(lambda x: CLASSES.index(x))

    df["is_bcc_train"] = df["is_bcc"]

    df["patient"] = df["patient_id"]
    df["split"] = df["site"] + "__" + df["patient"]

    df["sample_weight"] = 1  # np.where(df["is_bcc"] != -1, 1, 0.1)

    return df


def prepare_data_octiss(df):
    """
    Prepares the dataframe for the octiss data.

    Args:
        df (pandas dataframe): Test dataframe.

    Returns:
        pandas dataframe: Prepared dataframe.
    """
    df["site"] = ""
    df["img_id"] = df["img_name"].apply(lambda x: x.rsplit('_', 1)[0])
    df["frame"] = df["img_name"].apply(lambda x: int(x.rsplit('_', 1)[1]))

    df["img_name"] += ".png"

    nb_frames = (
        df.groupby(["img_id"])[["frame"]]
        .agg(lambda x: len(list(x)))
        .rename(columns={"frame": "nb_frames"})
    )
    df = df.merge(nb_frames, on=["img_id"])

    df["target"] = df["category"] % 3

    df["patient"] = df["patient_id"]
    df["split"] = df["patient_id"]

    df["sample_weight"] = 1

    return df
