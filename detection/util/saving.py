import pydicom
import skimage.io
import numpy as np


def load_stack(path, return_metadata=False):
    """
    Loads a .tif(f) or .dcm stack.

    Args:
        path (str): Path to stack.
        return_metadata (bool, optional): Whether to return the metadata. Defaults to False.

    Raises:
        NotImplementedError: Format not supported.

    Returns:
        np array: Stack.
        dict : Dicom metadata if return_metadata True.
    """
    metadata = None
    if path.lower().endswith(".dcm"):
        ds = pydicom.filereader.dcmread(path, stop_before_pixels=False)
        stack = np.array(ds.pixel_array)

        if return_metadata:
            metadata = pydicom.filereader.dcmread(path, stop_before_pixels=True)

    elif path.lower().endswith(".tiff") or path.lower().endswith(".tif"):
        stack = skimage.io.imread(path)
    else:
        raise NotImplementedError

    if len(stack.shape) == 3 and np.max(stack.shape) < 1800 and stack.shape[0] != 500:
        stack = np.swapaxes(stack, 0, 1)

    if return_metadata:
        return stack, metadata
    else:
        return stack
