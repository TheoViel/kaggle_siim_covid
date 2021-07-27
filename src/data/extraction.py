import gdcm  # noqa
import pydicom
import numpy as np


def read_xray(path):
    metadata = pydicom.read_file(path, stop_before_pixels=True)
    data = pydicom.read_file(path).pixel_array

    if metadata.PhotometricInterpretation == "MONOCHROME1":  # Inverted xray
        data = np.max(data) - data

    data = data - np.min(data)

    return data, metadata


def remove_padding(img):
    pad_y = (img > 30).mean(-1) > 0.1
    start_y = pad_y.tolist().index(True)
    end_y = pad_y[::-1].tolist().index(True)

    pad_x = (img > 30).mean(0) > 0.1
    start_x = pad_x.tolist().index(True)
    end_x = pad_x[::-1].tolist().index(True)

    if not np.any([start_x, end_x, start_y, end_y]):
        pad_y = (img < 255 - 30).mean(-1) > 0.1
        start_y = pad_y.tolist().index(True)
        end_y = pad_y[::-1].tolist().index(True)

        pad_x = (img < 255 - 30).mean(0) > 0.1
        start_x = pad_x.tolist().index(True)
        end_x = pad_x[::-1].tolist().index(True)

    img = img[start_y: len(img) - end_y, start_x: img.shape[1] - end_x]

    return img, (start_y, start_x)


def auto_windowing(img):
    pixels = img.flatten()
    pixels = pixels[pixels > 0]
    pixels = pixels[pixels < pixels.max()]
    start = np.percentile(pixels, 1)
    end = np.percentile(pixels, 99)

    img = np.clip(img, start, end)

    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)

    return img, (start, end)
