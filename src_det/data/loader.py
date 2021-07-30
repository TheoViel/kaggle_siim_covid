import torch


def collate_fn_train_yolo(batch):
    img, boxes, shapes = zip(*batch)

    labels = []
    for i, box in enumerate(boxes):
        if len(box):
            label = torch.zeros((len(box), 6))
            label[:, 2:] = torch.from_numpy(box)
            label[:, 0] = i
            labels.append(label)

    return torch.stack(list(img), 0), torch.cat(labels, 0), shapes


def collate_fn_val_yolo(batch):
    img, boxes, shapes = zip(*batch)
    return torch.stack(list(img), 0), boxes, shapes


def collate_fn_train_effdet(batch):
    img, boxes_, shapes = zip(*batch)

    boxes = []
    for b in boxes_:
        if len(b):
            b[:, [0, 1, 2, 3]] = b[:, [1, 0, 3, 2]]  # xyxy -> yxyx
        boxes.append(torch.from_numpy(b).float().view(-1, 4))

    # boxes = [torch.from_numpy(b).float().view(-1, 4) for b in boxes]
    return torch.stack(list(img), 0), boxes, shapes


def collate_fn_val_effdet(batch):
    img, boxes, shapes = zip(*batch)
    return torch.stack(list(img), 0), boxes, shapes


def get_collate_fns(selected_model):
    if "yolo" in selected_model:
        return collate_fn_train_yolo, collate_fn_val_yolo
    elif "efficientdet" in selected_model:
        return collate_fn_train_effdet, collate_fn_val_effdet
    else:
        raise NotImplementedError
