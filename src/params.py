import torch
import numpy as np

NUM_WORKERS = 4

DATA_PATH = "../input/"
TRAIN_IMG_PATH = DATA_PATH + "train/"
TEST_IMG_PATH = DATA_PATH + "train/"
LOG_PATH = "../logs/"
FEATURES_PATH = "../features/"

SIZE = 512
IMG_SIZE = (512, 512)

DCM_PATH = "../../../data/siim_covid/"
TRAIN_DCM_PATH = DCM_PATH + "train/"
TEST_DCM_PATH = DCM_PATH + "test/"

CLASSES = ["negative", "typical", "indeterminate", "atypical"]

NUM_CLASSES = len(CLASSES)

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
