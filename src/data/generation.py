import pydicom
import skimage.io
import numpy as np


def load_stack(path):
    """
    Loads a .tif(f) or .dcm stack.

    Args:
        path (str): Path to stack.

    Raises:
        NotImplementedError: Format not supported.

    Returns:
        np array: Stack
    """
    if path.endswith(".dcm"):
        ds = pydicom.filereader.dcmread(path, stop_before_pixels=False)
        stack = np.array(ds.pixel_array)

        metadata = pydicom.filereader.dcmread(path, stop_before_pixels=True)

    elif path.endswith(".tiff") or path.endswith(".tif"):
        stack = skimage.io.MultiImage(path)
        stack = np.array([i for i in stack])
        metadata = None
    else:
        raise NotImplementedError

    if len(stack.shape) == 3 and np.max(stack.shape) < 1800 and stack.shape[0] != 500:
        # Transpose first two axes for 3Ds
        stack = np.swapaxes(stack, 0, 1)

    if len(stack.shape) == 2:
        # Add first axis for 2Ds
        stack = stack[None, :]

    return stack, metadata


def detect_type(stack):
    """
    Detects the type of a stack using its shape.
    Can be "img", "vid" or "3d"

    Args:
        stack (np array): Stack.

    Returns:
        str: Stack type.
    """
    if len(stack) == 1:
        return "img"
    if 2048 in stack.shape:
        return "vid"
    else:
        return "3d"


def adapt_ratio(stack, max_ratio=4.75):
    """
    Processes 3D. Indicates if their ratio is abnormal.
    Crops or pad the image if it is reasonnable to do so.
    Thresholds are chosen visually.

    Args:
        stack (np array): Stack.
        max_ratio (float, optional): Maximum ratio. Defaults to 4.75.

    Returns:
        bool: Whether the stack is ok to use.
        np array: Processed stack.
    """
    z, x, y = stack.shape

    ratio = y / x
    third = y // 3  # perfect 1/3 ratio

    if (
        ratio > max_ratio or ratio < 1.5 or min(x, y) < 100
    ):  # Stack is abnormal, don't use it
        return False, stack

    elif (
        ratio > 3.3
    ):  # ratio is high, we fix this by padding to a 1/3 ratio on the bottom and top
        padding_bot = np.zeros((z, third - x, y))
        stack = np.concatenate((stack, padding_bot), 1)

    elif (
        ratio < 3
    ):  # Ratio is too low, we crop the bottom of the stack to a ratio of 1/3
        stack = stack[:, :third, :]

    else:  # Stack is kept as-is
        pass

    return True, stack


def map_label(label):
    """
    Function to map raw labels to the desired ones.

    Args:
        label (str): Raw label.

    Returns:
        str: Mappend label.
    """
    label = label.lower()  # noqa
    if "vs" in label or label == "":
        return "UNK"  # TODO ? : Consider multiple possiblities

    if "healthy" in label:
        return "HEALTHY"
    elif "bcc" in label or "bbc" in label:
        return "BCC"
    elif "scc" in label or "sdcc" in label or "spino" in label:
        return "SCC"
    elif "ak" in label or "ka" in label:
        return "AK"
    elif "bowen" in label:
        return "BOWEN"
    elif "vus" in label or "nid" in label or "sl" in label:
        return "NEVUS"
    elif "melanom" in label or "mm" in label or "lm" in label or "ssm" in label:
        return "MELANOMA"
    elif "lichen" in label or "psoriasis" in label or "eczema" in label:
        return "INFLAMATORY"
    elif "sk" in label or "ks" in label:
        return "SK"
    elif (
        "sebaceous_hyperplasia" in label
        or "sh" in label
        or "hyperplasie_sebacee" in label
    ):
        return "SH"

    else:
        #         print(l, '!')
        return "OTHER"


def treat_patient_2D(patient):
    """
    Processes a patient string in the 2D database.

    Args:
        patient (str): Patient string.

    Returns:
        str: Patient id.
        str: Acquisition.
        dict: Label associated to lesions.
    """
    patient_id = "_".join(patient.split("_")[:2])
    label = "_".join(patient.split("_")[2:-3])

    if "-" in patient_id:
        patient_id, acquisition = patient_id.split("-")
    else:
        acquisition = ""

    return patient_id, acquisition, {"*": label}


def annots_to_dict(annots):
    """
    Transforms a string that contains the annotation to a dictionary of {lesion : annotation}.

    Args:
        annots (str): Annotation string.

    Returns:
        dict: Label associated to lesions.
    """
    if len(annots) == 1:
        if not annots[0].startswith("L0"):
            return {"*": annots[0]}

    annot_dict = {}
    for annot in annots:
        lesions = []
        for i, a in enumerate(annot.split("_")):
            if a.startswith("L0"):
                lesions.append(a)
            elif a == "and":
                pass
            else:
                break
        label = "_".join(annot.split("_")[i:])

        for lesion in lesions:
            annot_dict[lesion] = label

    # Handles L0x0x keys
    new_annot_dict = {}
    for k in annot_dict.keys():
        if len(k) > 3:
            lesions = [f"L{int(lesion):02d}" for lesion in k.split("0")[1:]]
            for lesion in lesions:
                new_annot_dict[lesion] = annot_dict[k]
        else:
            new_annot_dict[k] = annot_dict[k]

    return new_annot_dict


def treat_patient_3D(patient):
    """
    Processes a patient string in the 3D database.

    Args:
        patient (str): Patient string.

    Returns:
        str: Patient id.
        str: Acquisition.
        dict: Label associated to lesions.
    """
    patient_id = "_".join(patient.split("_")[:2])

    annotations = "_".join(patient.split("_")[2:])
    while "study" in annotations:
        annotations = "_".join(annotations.split("_")[:-1])

    annots = []
    annot = ""
    for i, s in enumerate(annotations.split("_")):
        if "L0" in s:
            if annot:
                annots.append(annot[:-1])
                annot = ""
        annot += s + "_"
    annots.append(annot[:-1])

    # Handle "and"
    annots_join = [annots[0]]
    for i, annot in enumerate(annots):
        if i > 0:
            if annots_join[-1].split("_")[-1] == "and":
                annots_join[-1] += "_" + annot
            else:
                annots_join.append(annot)

    annots = annots_to_dict(annots_join)

    return patient_id, "", annots


def treat_infos(infos, zones):
    """
    Parses information containes in the subfolder string corresponding to a lesion.
    It looks for a zone containes in the zones list.
    The rest of the string is then the label of the lesion.

    Args:
        infos (str): Subfolder string.
        zones (list): Reference zones.

    Returns:
        str: Zone.
        str: Label.
    """
    infos = infos.split("_")
    for i in range(len(infos)):
        if "_".join(infos[i:]) in zones:
            break

    zone = "_".join(infos[i:])
    label = "_".join(infos[:i])
    return zone, map_label(label)


def is_bcc(label):
    """
    Indicates if a label is a bcc :
    BCC = 1, UNK = -1, others = 0.

    Args:
        label (str): Label string.

    Returns:
        int: Label is a bcc.
    """
    if label == "UNK":
        return -1
    elif label == "BCC":
        return 1
    else:
        return 0


def detect_glass_anomaly(img, threshold=13):
    """
    Tries to detect anomalies using the glass of the lense.

    Args:
        img (np array [H x W x C]): Image.
        threshold (int, optional): Threshold for detecting the glass. Defaults to 13.

    Returns:
        bool: Whether an anomaly was detected.
        int: height of the glass.
    """

    img = img.astype(float) / 255

    sums = img.sum(1)
    dsums = sums[1:] - sums[:-1]
    d2sums = np.abs(dsums[1:] - dsums[:-1])

    mask = (img.sum(1) > 0)[2:]
    d2sums *= mask

    glass = np.argmax(d2sums)
    max_ = np.max(d2sums)

    if max_ < threshold or (glass - (1 - mask).sum()) > (0.66 * len(sums)):
        return True, glass
    else:
        return False, glass


def detect_light_anomaly(img, glass=0, threshold=0.1):
    """
    Detects an anomaly in image brightness.

    Args:
        img (np array [H x W x C]): Image.
        glass (int, optional): Where the glass was detected. Defaults to 0.
        threshold (float, optional): Minimum brightness accepted. Defaults to 0.1.

    Returns:
        bool: Whether an anomaly was detected.
    """
    mask = img.sum(1) > 0
    brightness = (img[glass:, :].astype(float) / 255).mean() / mask.mean()

    if brightness < threshold:
        return True, brightness
    else:
        return False, brightness
