from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, BatchSampler

from utils.torch import worker_init_fn
from params import NUM_WORKERS


class PatientSampler(BatchSampler):
    """
    Custom PyTorch Sampler that limits the number of images per patient.
    Note that the limit can also be made on recordings,
    although the nomenclature was chosen for the patient case.
    """

    def __init__(
        self, sampler, patients, batch_size=32, drop_last=False, samples_per_patient=10
    ):
        """
        Constructor.

        Args:
            sampler (torch sampler): Initial sampler for the dataset, e.g. RandomSampler
            patients (numpy array): Patient corresponding to each sample. Precomputed to gain time.
            batch_size (int, optional): Batch size. Defaults to 32.
            drop_last (bool, optional): Whether to discard the last batch. Defaults to False.
            samples_per_patient (int, optional): Maximum of image to use per id. Defaults to 10.
        """
        super().__init__(sampler, batch_size, drop_last)
        self.samples_per_patient = samples_per_patient
        self.patients = patients

        self.len = self.compute_len()

    def __len__(self):
        return self.len

    def compute_len(self):
        patient_counts = {}
        yielded = 0
        batch = []

        for idx in self.sampler:
            patient = self.patients[idx]
            try:
                patient_counts[patient] += 1
            except KeyError:
                patient_counts[patient] = 1

            if patient_counts[patient] <= self.samples_per_patient:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yielded += 1
                    batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1

        return yielded

    def __iter__(self):
        """
        Iterator.
        Only adds an index to a batch if the associated patients has not be sampled too many time.

        Yields:
            torch tensors : batches.
        """
        patient_counts = {}
        yielded = 0
        batch = []

        for idx in self.sampler:
            patient = self.patients[idx]
            try:
                patient_counts[patient] += 1
            except KeyError:
                patient_counts[patient] = 1

            if patient_counts[patient] <= self.samples_per_patient:
                batch.append(idx)

                if len(batch) == self.batch_size:
                    yield batch
                    yielded += 1
                    batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch


def define_loaders(train_dataset, val_dataset, samples_per_patient=0, batch_size=32, val_bs=32):
    """
    Builds data loaders.

    Args:
        train_dataset (ColorBCCDataset): Dataset to train with.
        val_dataset (ColorBCCDataset): Dataset to validate with.
        samples_per_patient (int, optional): Number of images to use per patient. Defaults to 0.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.

    Returns:
       DataLoader: Train loader.
       DataLoader: Val loader.
    """
    if samples_per_patient:
        sampler = PatientSampler(
            RandomSampler(train_dataset),
            train_dataset.patients_img,
            batch_size=batch_size,
            drop_last=True,
            samples_per_patient=samples_per_patient,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

    print(
        f"Using {len(train_loader)} out of {len(train_dataset) // batch_size} "
        f"batches by limiting to {samples_per_patient} samples per patient.\n"
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader
