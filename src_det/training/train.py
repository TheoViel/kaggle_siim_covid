import time
import torch

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from params import NUM_WORKERS
from data.loader import get_collate_fns
from training.meter import DetectionMeter
from training.optim import define_optimizer


def fit(
    model,
    config,
    train_dataset,
    val_dataset,
):
    """
    Usual torch fit function.

    Args:
        model (torch model): Model to train.
        config (Config): Parameters.
        train_dataset (torch dataset): Dataset to train with.
        val_dataset (torch dataset): Dataset to validate with.

    Returns:
        numpy array [len(val_dataset) x num_classes]: Last prediction on the validation data.
        pandas dataframe: Training history.
    """
    optimizer = define_optimizer(model, config)

    collate_fn_train, collate_fn_val = get_collate_fns(config.selected_model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_train,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_bs,
        shuffle=False,
        collate_fn=collate_fn_val,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    meter = DetectionMeter(
        pred_format=config.pred_format, truth_format=config.bbox_format
    )

    num_warmup_steps = int(config.warmup_prop * config.epochs * len(train_loader))
    num_training_steps = int(config.epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(config.epochs):
        model.train()

        start_time = time.time()
        optimizer.zero_grad()

        avg_loss = 0
        for batch in train_loader:
            x = batch[0].to(config.device)
            y_batch = batch[1]

            y_pred = model(x)

            loss = model.compute_loss(y_pred, y_batch)
            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            scheduler.step()

            for param in model.parameters():
                param.grad = None

        model.eval()
        meter.reset()
        if epoch + 1 >= config.first_epoch_eval:
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(config.device)
                    pred_boxes = model(x)
                    meter.update(batch[1], pred_boxes, shapes=x.size(), images=batch[0])
            metrics = meter.compute()

        elapsed_time = time.time() - start_time
        if (epoch + 1) % config.verbose == 0:
            elapsed_time = elapsed_time * config.verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{config.epochs:02d} \t lr={lr:.1e}\t "
                f"t={elapsed_time:.0f}s \t loss={avg_loss:.3f}",
                end="\t",
            )
            if epoch + 1 >= config.first_epoch_eval:
                print(f"f1={metrics['f1_score']:.3f} \t recall={metrics['recall']:.3f} \t")
                if config.verbose_plot:
                    if (epoch + 1) % config.verbose_plot == 0:
                        meter.plot(n_samples=3)
            else:
                print("")

    del val_loader, train_loader, y_pred, loss, x, y_batch
    torch.cuda.empty_cache()

    return meter
