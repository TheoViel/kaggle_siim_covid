import torch
import numpy as np

NUM_WORKERS = 4

DATA_PATH = "../data/"
LOG_PATH = "../logs/"
OUT_DIR = "../output/"

SIZE = (256, 512)  # (320, 640)
ORIG_SIZE = (500, 1232)
IMG_SIZE = np.mean(SIZE)

IMG_PATH = DATA_PATH + 'processed/'

METADATA_PATHS = [
    'metadata_cheflow.csv',
    'metadata_octatris.csv',
    'metadata_octatris_t3.csv',
]

NUM_CLASSES = 1

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
