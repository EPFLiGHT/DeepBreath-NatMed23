import numpy as np
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler


def balance_weights(weight_a, weight_b, alpha):
    assert 0 <= alpha <= 1, f"invalid alpha {alpha}, must be 0 <= alpha <= 1"

    beta = 1 - alpha
    weights = (alpha * weight_a) + (beta * weight_b)

    return weights


class BalancedSampler:
    """
    Factory class to create balanced samplers.
    """

    def __init__(self, df, pos_label, label_col="diagnosis", additional_cols=[]):
        by = [label_col] + additional_cols
        grouped_samples = (
            df.reset_index().groupby(by)["index"].agg(["size", list]).reset_index()
        )

        grouped_samples["original_weight"] = (
            grouped_samples["size"] / grouped_samples["size"].sum()
        )

        pos_indices = (grouped_samples[label_col].isin(pos_label)).values
        n_pos = pos_indices.sum()
        n_neg = len(pos_indices) - n_pos
        grouped_samples["uniform_weight"] = np.zeros(len(grouped_samples))
        grouped_samples.loc[pos_indices, "uniform_weight"] = 0.5 / n_pos
        grouped_samples.loc[~pos_indices, "uniform_weight"] = 0.5 / n_neg

        self.grouped_samples = grouped_samples

    def _weight_classes(self, class_idxs, alpha):
        class_sizes = np.asarray([len(idxs) for idxs in class_idxs])

        original_weights = self.grouped_samples.original_weight.values
        uniform_weights = self.grouped_samples.uniform_weight.values

        weights = balance_weights(uniform_weights, original_weights, alpha)

        return class_sizes, weights

    def get_sampler(self, batch_size, n_batches=None, alpha=0.5):
        """
        Parameters
        ----------
        class_idxs : 2D list of ints
            List of sample indices for each class. Eg. [[0, 1], [2, 3]] implies indices 0, 1
            belong to class 0, and indices 2, 3 belong to class 1.

        batch_size : int
            The batch size to use.

        n_batches : int
            The number of batches per epoch.

        alpha : numeric in range [0, 1]
            Weighting term used to determine weights of each class in each batch.
            When `alpha` == 0, the batch class distribution will approximate the training population
            class distribution.
            When `alpha` == 1, the batch class distribution will approximate a uniform distribution,
            with equal number of samples from each class.
        """
        n_samples = self.grouped_samples["size"].sum()
        class_idxs = self.grouped_samples["list"].values.tolist()
        class_sizes, weights = self._weight_classes(class_idxs, alpha)
        sample_rates = weights / class_sizes
        n_batches = n_batches if n_batches is not None else n_samples // batch_size

        print("\n", "Sampling Statistics:")
        df_to_print = self.grouped_samples.drop(
            columns=["list", "original_weight", "uniform_weight"]
        )
        expected_frequencies = sample_rates * self.grouped_samples["size"]
        expected_frequencies /= expected_frequencies.sum()
        df_to_print["expected_samples"] = (n_samples * expected_frequencies).astype(
            np.int
        )
        df_to_print["expected_frequency"] = expected_frequencies
        print(df_to_print, "\n")

        return WeightedRandomBatchSampler(
            sample_rates, class_idxs, batch_size, n_batches
        )


class WeightedRandomBatchSampler(BatchSampler):
    """
    Samples with replacement according to the provided weights.

    Parameters
    ----------
    class_weights : `numpy.array(int)`
        The number of samples of each class to include in each batch.

    class_idxs : 2D list of ints
        The indices that correspond to samples of each class.

    batch_size : int
        The size of each batch yielded.

    n_batches : int
        The number of batches to yield.
    """

    def __init__(self, class_weights, class_idxs, batch_size, n_batches):
        self.sample_idxs = []
        for idxs in class_idxs:
            self.sample_idxs.extend(idxs)

        sample_weights = []
        for c, weight in enumerate(class_weights):
            sample_weights.extend([weight] * len(class_idxs[c]))

        self.sampler = WeightedRandomSampler(
            sample_weights, batch_size, replacement=True
        )
        self.n_batches = n_batches

    def __iter__(self):
        for bidx in range(self.n_batches):
            selected = []
            for idx in self.sampler:
                selected.append(self.sample_idxs[idx])
            yield selected

    def __len__(self):
        return self.n_batches
