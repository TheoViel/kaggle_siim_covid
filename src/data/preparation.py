import cv2
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from utils.boxes import treat_boxes, Boxes, merge_boxes
from params import DATA_PATH, SIZE, CLASSES
from utils.plot import plot_sample


def prepare_dataframe():
    df_study = pd.read_csv(DATA_PATH + 'train_study_level.csv')
    df_image = pd.read_csv(DATA_PATH + 'train_image_level.csv')
    df = pd.read_csv(DATA_PATH + f'df_train_{SIZE}.csv')
    folds = pd.read_csv(DATA_PATH + 'covid_folds.csv')

    df_study['study_id'] = df_study['id'].apply(lambda x: x.split('_')[0])
    df_study = df_study.rename(columns={c: c.split(' ')[0].lower() for c in df_study.columns})
    df_study.drop('id', axis=1, inplace=True)

    df_image = df_image.rename(columns={'id': "image_id", "StudyInstanceUID": "study_id"})
    df_image['image_id'] = df_image['image_id'].apply(lambda x: x.split('_')[0])
    df_image['boxes'] = df_image['boxes'].apply(treat_boxes)
    df_image['label'] = df_image['label'].apply(lambda x: x.split(' ')[0])
    df_image['img_target'] = (df_image['label'] == "opacity").astype(int)

    df['shape'] = df['shape'].apply(lambda x: np.array(x[1:-1].split(', ')).astype(int))
    df['shape_crop'] = df['shape_crop'].apply(lambda x: np.array(x[1:-1].split(', ')).astype(int))
    df['crop_starts'] = df['crop_starts'].apply(lambda x: np.array(x[1:-1].split(', ')).astype(int))

    df_image = df_image.merge(df_study, on="study_id", how="left")
    df = df.merge(df_image, on=['study_id', 'image_id'])
    df['study_label'] = [CLASSES[c] for c in df[CLASSES].values.argmax(-1)]

    folds['image_id'] = folds['image_id'].apply(lambda x: x.split('_')[0])
    folds = folds[['image_id', 'kfold']]
    df = df.merge(folds, on='image_id', how='left')

    return df.reset_index(drop=True)


def handle_duplicates(df, clusts, transpositions, plot=False):
    root = DATA_PATH + f"train_{SIZE}/"

    for clust, tran in zip(clusts, transpositions):
        df_clust = df[df['save_name'].apply(lambda x: np.isin(x, clust))].copy()

        boxes = [
            Boxes(np.array(box), shape, bbox_format="coco")
            for box, shape in df_clust[['boxes', 'shape']].values
        ]

        count = Counter(df_clust['label'])
        num_shapes = len(df_clust['shape'].apply(lambda x: " ".join(x.astype(str))).unique())
        assert num_shapes == 1, "Different img shapes, warning !"

        fused_boxes = merge_boxes(boxes, tran)

        for idx, boxes in zip(df_clust.index,  fused_boxes):
            df.at[idx, 'boxes'] = boxes["coco"].tolist()

        if plot:
            print(f'Clust {clust}')
            print(f'Trans {tran}')
            print(count)

            plt.figure(figsize=(15, (len(clust) // 3 + 1) * 5))
            for i, n in enumerate(clust):
                plt.subplot(len(clust) // 3 + 1, 3, i + 1)
                img = cv2.imread(root + n)
                plot_sample(img, fused_boxes[i]["yolo"])
                plt.axis(False)
            plt.show()

    df['label'] = df['boxes'].apply(lambda x: "opacity" if len(x) else "none")
    df['img_target'] = (df['label'] == "opacity").astype(int)
    return df


def add_additional_boxes(df):
    # additional_boxes = pd.read_csv(DATA_PATH + "additional_boxes_.csv", sep=";")

    # additional_boxes['boxes'] = (
    #     additional_boxes['x'].astype(str) + ', ' +
    #     additional_boxes['y'].astype(str) + ', ' +
    #     additional_boxes['w'].astype(str) + ', ' +
    #     additional_boxes['h'].astype(str)
    # )
    # additional_boxes['boxes'] = additional_boxes['boxes'].apply(lambda x: x.split(', '))
    # additional_boxes.drop(['shape_0', 'shape_1', 'x', 'y', 'w', 'h'], axis=1, inplace=True)

    # additional_boxes = additional_boxes.groupby('save_name').agg(list).reset_index()
    # additional_boxes['label'] = additional_boxes['label'].apply(lambda x: x[0])

    # additional_boxes.to_csv('../output/additional_boxes_processed.csv', index=False)
    additional_boxes = pd.read_csv('../output/additional_boxes_processed.csv')

    # process boxes
    additional_boxes['boxes'] = additional_boxes['boxes'].apply(
        lambda x: np.array(ast.literal_eval(x)).astype(int).tolist()
    )

    # merge df
    df = df.merge(additional_boxes, how='left', on='save_name', suffixes=['', '_add'])

    # Remove outlier
    df = df[df['label_add'] != "remove"].reset_index(drop=True)

    # Merge boxes
    for row in df.loc[df.boxes_add.isnull(), 'boxes_add'].index:
        df.at[row, 'boxes_add'] = []
    df['boxes'] = df['boxes'] + df['boxes_add']

    # Merge label
    df['img_target'] = np.clip(df['img_target'] + (df['label_add'] == "opacity").astype(int), 0, 1)
    df['label'] = df['img_target'].apply(lambda x: 'opacity'if x else 'none')

    # Remove useless columns
    df.drop(['boxes_add', 'label_add'], axis=1, inplace=True)
    # df = df.dropna().reset_index()  # for debugging

    return df
