import time
import torch
import numpy as np
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup

from utils.metrics import per_class_average_precision_score, study_level_map
from training.loader import define_loaders
from training.mix import cutmix_data
from training.losses import CovidLoss
from training.optim import RAdam, Lookahead


ONE_HOT = np.eye(10)


def fit(
    model,
    train_dataset,
    val_dataset,
    samples_per_patient=0,
    optimizer_name="adam",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    mix="cutmix",
    mix_proba=0.0,
    mix_alpha=1.0,
    num_classes=1,
    verbose=1,
    first_epoch_eval=0,
    device="cuda"
):
    """
    Usual torch fit function.

    Args:
        model (torch model): Model.
        train_dataset (ColorBCCDataset): Dataset to train with.
        val_dataset (ColorBCCDataset): Dataset to validate with.
        samples_per_patient (int, optional): Number of images to use per patient. Defaults to 0.
        optimizer_name (str, optional): Optimizer name. Defaults to 'adam'.
        epochs (int, optional): Number of epochs. Defaults to 50.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
        warmup_prop (float, optional): Warmup proportion. Defaults to 0.1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        mix (str, optional): Mixing strategy (either mixup or cutmix). Defaults to "cutmix".
        mix_proba (float, optional): Probability to apply mixing with. Defaults to 0..
        mix_alpha (float, optional): Alpha parameter for mixing. Defaults to 1..
        num_classes (int, optional): Number of classes. Defaults to 1.
        verbose (int, optional): Period (in epochs) to display logs at. Defaults to 1.
        first_epoch_eval (int, optional): Epoch to start evaluating at. Defaults to 0.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(val_dataset) x num_classes]: Last prediction on the validation data.
        pandas dataframe: Training history.
    """
    avg_val_loss = 0.
    lam = 1

    # Optimizer
    if optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        optimizer = Lookahead(
            RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, use_gc=True),
            alpha=0.5,
            k=5
        )

    scaler = torch.cuda.amp.GradScaler()

    # Data loaders
    train_loader, val_loader = define_loaders(
        train_dataset,
        val_dataset,
        samples_per_patient=samples_per_patient,
        batch_size=batch_size,
        val_bs=val_bs
    )

    # Losses
    loss_fct = CovidLoss()

    # Scheduler
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        optimizer.zero_grad()

        avg_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            masks = batch[1].to(device)
            y_study = batch[2].to(device)
            y_img = batch[3].to(device)

            # Mix data
            apply_mix = np.random.rand() < mix_proba
            if apply_mix:
                x, y_study, y_img, masks, lam = cutmix_data(
                    x, y_study, y_img, masks, alpha=mix_alpha, device=device
                )

            with torch.cuda.amp.autocast():
                # Forward
                pred_study, pred_img, preds_mask = model(x)

                # Compute losses
                loss = loss_fct(
                    pred_study, pred_img, preds_mask, y_study, y_img, masks, mix_lambda=lam
                ).mean()

                # Backward & parameter update
                scaler.scale(loss).backward()
                avg_loss += loss.item() / len(train_loader)

                scaler.step(optimizer)

                scale = scaler.get_scale()
                scaler.update()

                if scale == scaler.get_scale():
                    scheduler.step()

                for param in model.parameters():
                    param.grad = None

        model.eval()
        avg_val_loss = 0.
        study_ap, img_ap = 0, 0
        preds_study = np.empty((0, num_classes))
        preds_img = np.empty((0))

        if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == epochs):
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    masks = batch[1].to(device)
                    y_study = batch[2].to(device)
                    y_img = batch[3].to(device)

                    # Forward
                    pred_study, pred_img, preds_mask = model(x)

                    # Compute losses
                    loss = loss_fct(
                        pred_study.detach(),
                        pred_img.detach(),
                        [p.detach() for p in preds_mask],
                        y_study,
                        y_img,
                        masks,
                    ).mean()
                    avg_val_loss += loss.mean().item() / len(val_loader)

                    # Update predictions
                    pred_study = torch.softmax(pred_study, -1)
                    pred_img = torch.sigmoid(pred_img).view(-1)
                    preds_study = np.concatenate([preds_study, pred_study.cpu().numpy()])
                    preds_img = np.concatenate([preds_img, pred_img.cpu().numpy()])

                study_ap = study_level_map(
                    preds_study, val_dataset.study_targets, val_dataset.studies
                )
                img_ap = per_class_average_precision_score(preds_img, val_dataset.img_targets)

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s \t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )

            if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == epochs):
                print(
                    f"val_loss={avg_val_loss:.3f}\t study_map={study_ap:.3f}\t img_map={img_ap:.3f}"
                )
            else:
                print("")

    del (val_loader, train_loader, loss, x, masks, y_study, y_img, pred_study, pred_img, preds_mask)
    torch.cuda.empty_cache()

    return preds_study, preds_img
