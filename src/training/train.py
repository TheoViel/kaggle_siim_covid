import time
import torch
import numpy as np

from torchcontrib.optim import SWA
from transformers import get_linear_schedule_with_warmup


from utils.logger import update_history
from utils.metrics import compute_metrics
from training.loader import define_loaders
from training.mix import cutmix_data, mixup_data
from training.optim import define_loss, prepare_for_loss, define_optimizer, ConsistencyLoss


ONE_HOT = np.eye(10)


def update_teacher_params(student, teacher, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)


def fit(
    model,
    mean_teacher,
    mean_teacher_config,
    train_dataset,
    val_dataset,
    samples_per_patient=0,
    optimizer_name="adam",
    loss_name="BCEWithLogitsLoss",
    activation="sigmoid",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    swa_first_epoch=50,
    mix="cutmix",
    mix_proba=0.0,
    mix_alpha=1.0,
    num_classes=1,
    aux_loss_weight=0.1,
    verbose=1,
    first_epoch_eval=0,
    device="cuda"
):
    """
    Usual torch fit function.
    Supports cutmix, mixup & SWA.

    Args:
        model (torch model): Student model.
        mean_teacher (torch model or None): Teacher model. If None then mean teacher is not used.
        mean_teacher_config (dict or None): Mean teacher training parameters.
        train_dataset (ColorBCCDataset): Dataset to train with.
        val_dataset (ColorBCCDataset): Dataset to validate with.
        samples_per_patient (int, optional): Number of images to use per patient. Defaults to 0.
        optimizer_name (str, optional): Optimizer name. Defaults to 'adam'.
        loss_name (str, optional): Loss name. Defaults to 'BCEWithLogitsLoss'.
        activation (str, optional): Activation function. Defaults to 'sigmoid'.
        epochs (int, optional): Number of epochs. Defaults to 50.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
        warmup_prop (float, optional): Warmup proportion. Defaults to 0.1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        swa_first_epoch (int, optional): Epoch to start applying SWA from. Defaults to 50.
        mix (str, optional): Mixing strategy (either mixup or cutmix). Defaults to "cutmix".
        mix_proba (float, optional): Probability to apply mixing with. Defaults to 0..
        mix_alpha (float, optional): Alpha parameter for mixing. Defaults to 1..
        num_classes (int, optional): Number of classes. Defaults to 1.
        aux_loss_weight (float, optional): Weight for the auxiliary loss. Defaults to 0.1.
        verbose (int, optional): Period (in epochs) to display logs at. Defaults to 1.
        first_epoch_eval (int, optional): Epoch to start evaluating at. Defaults to 0.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(val_dataset) x num_classes]: Last prediction on the validation data.
        pandas dataframe: Training history.
    """
    global_step = 1
    avg_val_loss = 0.0
    history = None
    mix_fct = cutmix_data if mix == "cutmix" else mixup_data

    # Optimizer
    optimizer = define_optimizer(optimizer_name, model.parameters(), lr=lr)
    if swa_first_epoch <= epochs:
        optimizer = SWA(optimizer)

    # Data loaders
    train_loader, val_loader = define_loaders(
        train_dataset,
        val_dataset,
        samples_per_patient=samples_per_patient,
        batch_size=batch_size,
        val_bs=val_bs
    )

    # Losses
    loss_fct = define_loss(loss_name, device=device)

    aux_loss_name = "CrossEntropyLoss"
    loss_fct_aux = define_loss(aux_loss_name, device=device)

    if mean_teacher_config is not None:
        consistency_loss_fct = ConsistencyLoss(
            mean_teacher_config, activation=activation, steps_per_epoch=len(train_loader)
        )
        consistency_loss_fct_aux = ConsistencyLoss(
            mean_teacher_config, activation="softmax", steps_per_epoch=len(train_loader)
        )
    else:
        y_pred_teach, y_pred_aux_teach = None, None

    # Scheduler
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        if mean_teacher is not None:
            mean_teacher.train()

        start_time = time.time()
        optimizer.zero_grad()

        avg_loss = 0

        if epoch + 1 > swa_first_epoch:
            optimizer.swap_swa_sgd()
            # print("Swap to SGD")

        for batch in train_loader:
            x = batch[0].to(device)
            x_teach = batch[1].to(device)
            y_batch = batch[2].to(device)
            y_batch_aux = batch[3].to(device)
            sample_weight = batch[4].to(device)

            # Mix data
            apply_mix = np.random.rand() < mix_proba
            if apply_mix:
                x, x_teach, y_a, y_b, y_aux_a, y_aux_b, w_a, w_b, lam = mix_fct(
                    x, x_teach, y_batch, y_batch_aux, sample_weight, alpha=mix_alpha, device=device
                )
                y_batch = torch.clamp(y_a + y_b, 0, 1)
                y_batch_aux = lam * y_aux_a + (1 - lam) * y_aux_b
                sample_weight = w_a * lam + (1 - lam) * w_b
            else:
                y_a, y_b, y_aux_a, y_aux_b = None, None, None, None

            # Forward
            y_pred, y_pred_aux = model(x)
            if mean_teacher is not None:
                y_pred_teach, y_pred_aux_teach = mean_teacher(x)

            # Compute losses
            [y_pred, y_pred_teach], [y_batch, y_a, y_b] = prepare_for_loss(
                [y_pred, y_pred_teach], [y_batch, y_a, y_b], loss_name, device=device
            )
            [y_pred_aux, y_pred_aux_teach], [y_batch_aux, y_aux_a, y_aux_b] = prepare_for_loss(
                [y_pred_aux, y_pred_aux_teach],
                [y_batch_aux, y_aux_a, y_aux_b],
                aux_loss_name,
                device=device,
            )

            if apply_mix and loss_name == "CrossEntropyLoss":
                loss = lam * loss_fct(y_pred, y_a) + (1 - lam) * loss_fct(y_pred, y_b)
            else:
                loss = loss_fct(y_pred, y_batch)
            if len(loss.size()) == 2:
                loss = loss.mean(-1)

            if mean_teacher is not None:
                loss += consistency_loss_fct(y_pred, y_pred_teach, global_step)

            if aux_loss_weight:  # NOT CHECKED
                if apply_mix and aux_loss_name == "CrossEntropyLoss":
                    loss = lam * loss_fct(y_pred, y_a) + (1 - lam) * loss_fct(y_pred, y_b)
                else:
                    loss += aux_loss_weight * (
                        lam * loss_fct(y_pred_aux, y_aux_a) +
                        (1 - lam) * loss_fct(y_pred_aux, y_aux_b)
                    )

                if mean_teacher is not None:
                    loss += aux_loss_weight * consistency_loss_fct_aux(
                        y_pred_aux, y_pred_aux_teach, global_step
                    )

            loss = (loss * sample_weight).mean()

            # Backward & parameter update
            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            scheduler.step()

            if mean_teacher is not None:
                update_teacher_params(
                    model, mean_teacher, mean_teacher_config['ema_decay'], global_step
                )
            global_step += 1

            for param in model.parameters():
                param.grad = None

            if mean_teacher is not None:
                for param in mean_teacher.parameters():
                    param.grad = None

        if epoch + 1 >= swa_first_epoch:
            # print("update + swap to SWA")
            optimizer.update_swa()
            optimizer.swap_swa_sgd()

        model.eval()
        if mean_teacher is not None:
            mean_teacher.eval()
        avg_val_loss = 0.0
        metrics = compute_metrics(0, 0, num_classes=num_classes, dummy=True)

        if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == epochs):
            with torch.no_grad():
                preds = np.empty((0, num_classes))

                for batch in val_loader:
                    x = batch[0].to(device)
                    y_batch = batch[2].to(device)
                    y_batch_aux = batch[3].to(device)

                    # Forward
                    y_pred, y_pred_aux = model(x)

                    # Compute losses
                    [y_pred], [y_batch] = prepare_for_loss(
                        [y_pred], [y_batch], loss_name, device=device
                    )
                    [y_pred_aux], [y_batch_aux] = prepare_for_loss(
                        [y_pred_aux], [y_batch_aux], aux_loss_name, device=device
                    )

                    loss = loss_fct(y_pred, y_batch)
                    if aux_loss_weight:
                        loss += aux_loss_weight * loss_fct_aux(y_pred_aux, y_batch_aux)
                    avg_val_loss += loss.mean().item() / len(val_loader)

                    # Get probabilities
                    if activation == "sigmoid":
                        y_pred = torch.sigmoid(y_pred).view(-1, num_classes)
                    elif activation == "softmax":
                        y_pred = torch.softmax(y_pred, -1)

                    preds = np.concatenate([preds, y_pred.cpu().numpy()])

                metrics = compute_metrics(
                    preds, val_dataset.targets, num_classes=num_classes, loss_name=loss_name,
                )

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
                if loss_name != "CrossEntropyLoss":
                    print(f"val_loss={avg_val_loss:.3f} \t auc={metrics['auc'][0]:.3f}")
                else:
                    print(f"val_loss={avg_val_loss:.3f} \t acc={metrics['accuracy'][0]:.3f}")
            else:
                print("")
            history = update_history(
                history, metrics, epoch + 1, avg_loss, avg_val_loss, elapsed_time
            )

    if mix_proba:
        del (y_a, y_b, y_aux_a, y_aux_b, w_a, w_b,)
    del (val_loader, train_loader, y_pred, y_pred_aux, loss, x, y_batch, y_batch_aux, sample_weight)
    torch.cuda.empty_cache()

    return preds, history
