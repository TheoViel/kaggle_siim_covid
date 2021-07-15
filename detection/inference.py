import torch
import numpy as np
from torch.utils.data import DataLoader

from params import NUM_WORKERS
from data.loader import get_collate_fns
from training.meter import DetectionMeter

import sys
sys.path.append('../yolov5/')
from utils.general import non_max_suppression  # noqa


def predict(model, dataset, config):

    collate_fn_eval = get_collate_fns(config.selected_model)[-1]

    loader = DataLoader(
        dataset,
        batch_size=config.val_bs,
        shuffle=False,
        collate_fn=collate_fn_eval,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    meter = DetectionMeter(
        np.ones(len(dataset)), pred_format=config.pred_format, truth_format=config.bbox_format
    )

    meter.reset()
    model.eval()

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(config.device)
            pred_boxes = model(x)
            meter.update(batch[1], pred_boxes, shapes=x.size(), images=batch[0])

    meter.fusion()
    return meter
