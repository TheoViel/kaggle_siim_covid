import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)


ONE_HOT = np.eye(10)


class SeededGroupKFold(GroupKFold):
    """
    Extends sklearn's GroupKFold with to have a random_state argument.
    Allows for the use of different splits to test robustness.
    """

    def __init__(self, n_splits=5, random_state=0):
        """
        Constructor

        Args:
            n_splits (int, optional): Number of folds. Defaults to 5.
            random_state (int, optional): Seed. Defaults to 0.
        """
        super().__init__(n_splits)
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """
        Overwrites the split function of GroupKFold to add randomness.

        Args:
            X (np array): Data.
            y (np array, optional): Labels. Defaults to None.
            groups (np array, optional): Groups. Defaults to None.

        Yields:
            numpy array: Train indices.
            test array: Test indices.
        """
        assert groups is not None
        groups = pd.Series(groups)

        ix = np.arange(len(X))

        unique = np.unique(groups)
        np.random.RandomState(self.random_state).shuffle(unique)

        for split in np.array_split(unique, self.n_splits):
            mask = groups.isin(split)
            train, test = ix[~mask], ix[mask]
            yield train, test


def compute_metrics(pred, truth, num_classes=1, threshold=0.5, dummy=False, loss_name=""):
    """
    Computes metrics for the problem.

    Args:
        pred (numpy array): Predictions.
        truth (numpy array): Truths.
        num_classes (int, optional): Number of classes. Defaults to 1.
        dummy (bool, optional): Whether to return a placeholder. Defaults to False.
        loss_name (str, optional): Name of the loss, used for the choice of metrics. Defaults to "".

    Returns:
        dict : Metrics dictionary.
    """

    metrics = {
        "auc": [0],
        "accuracy": [0],
        "balanced_accuracy": [0],
        "f1": [0],
        "conf_mat": [np.zeros((num_classes, num_classes))],
    }

    if dummy:
        return metrics

    if num_classes == 1:  # binary
        auc = roc_auc_score(truth, pred)
        metrics["auc"] = [auc]

        accuracy = accuracy_score(truth, pred > 0.5)
        metrics["accuracy"] = [accuracy]

        f1 = f1_score(truth, pred > 0.5)
        metrics["f1"] = [f1]

        conf_mat = confusion_matrix(truth, pred > 0.5)
        metrics["conf_mat"] = [conf_mat]

    elif loss_name != "CrossEntropyLoss":  # Multilabel
        try:
            auc = np.mean(
                [roc_auc_score(truth[:, i], pred[:, i]) for i in range(num_classes)]
            )
        except ValueError:
            auc = 0
        metrics["auc"] = [auc]
    else:
        if len(truth.shape) == 1:
            truth = ONE_HOT[truth][:, :num_classes]

        try:
            auc = np.mean(
                [roc_auc_score(truth[:, i], pred[:, i]) for i in range(num_classes)]
            )
        except ValueError:
            auc = 0
        metrics["auc"] = [auc]

        accuracy = accuracy_score(truth.argmax(-1), pred.argmax(-1))
        metrics["accuracy"] = [accuracy]

        balanced_accuracy = balanced_accuracy_score(truth.argmax(-1), pred.argmax(-1))
        metrics["balanced_accuracy"] = [balanced_accuracy]

        f1 = f1_score(
            truth.argmax(-1), pred.argmax(-1), average="macro", zero_division=0
        )
        metrics["f1"] = [f1]

        conf_mat = confusion_matrix(np.argmax(truth, -1), np.argmax(pred, -1))
        metrics["conf_mat"] = [conf_mat]

    return metrics
