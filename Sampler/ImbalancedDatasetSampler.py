from typing import Callable
import torch

import Constants
from preprocessing.utils import class_distribution_counter, pickle_load


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
            self,
            dataset,
            labels: list = None,
            indices: list = None,
            num_samples: int = None,
            callback_get_label: Callable = None,
            device: str = 'cpu',
            **kwargs
    ):

        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        # df["label"] = self._get_labels(dataset) if labels is None else labels
        label_to_count = class_distribution_counter(**kwargs)

        go_terms = pickle_load(Constants.ROOT + "/go_terms")
        terms = go_terms['GO-terms-{}'.format(kwargs['ont'])]

        class_weights = [label_to_count[i] for i in terms]
        total = sum(class_weights)
        self.weights = torch.tensor([1.0 / label_to_count[i] for i in terms],
                                    dtype=torch.float).to(device)

    # def _get_labels(self, dataset):
    #     if self.callback_get_label:
    #         return self.callback_get_label(dataset)
    #     elif isinstance(dataset, torch.utils.data_bp.TensorDataset):
    #         return dataset.tensors[1]
    #     elif isinstance(dataset, torchvision.datasets.MNIST):
    #         return dataset.train_labels.tolist()
    #     elif isinstance(dataset, torchvision.datasets.ImageFolder):
    #         return [x[1] for x in dataset.imgs]
    #     elif isinstance(dataset, torchvision.datasets.DatasetFolder):
    #         return dataset.samples[:][1]
    #     elif isinstance(dataset, torch.utils.data_bp.Subset):
    #         return dataset.dataset.imgs[:][1]
    #     elif isinstance(dataset, torch.utils.data_bp.Dataset):
    #         return dataset.get_labels()
    #     else:
    #         raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

#
# from torch_geometric.loader import DataLoader
# from Dataset.Dataset import load_dataset
#
# kwargs = {
#     'seq_id': 0.95,
#     'ont': 'cellular_component',
#     'session': 'train'
# }
#
# dataset = load_dataset(root=Constants.ROOT, **kwargs)
# train_dataloader = DataLoader(dataset,
#                               batch_size=30,
#                               drop_last=False,
#                               sampler=ImbalancedDatasetSampler(dataset, **kwargs))
#
#
# for i in train_dataloader:
#     print(i)