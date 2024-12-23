import os
from copy import deepcopy
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
import torchvision

from mean_accumulator import MeanAccumulator
from .utils import DistributedSampler
from . import cifar_architectures
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

class Batch:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class CifarTask:
    def __init__(self, device, timer, architecture, seed, ds, bn = True):
        self._device = device
        self._timer = timer
        self._seed = seed
        self._architecture = architecture
        self.ds = ds
        self.bn = bn

        self._train_set, self._test_set = self._create_dataset(
            data_root = "./default_data_path"
        )

        self._model = self._create_model()
        self._criterion = torch.nn.CrossEntropyLoss().to(self._device)

        self._epoch = 0  # Counts how many times train_iterator was called



    @property
    def state(self):
        return [parameter for parameter in self._model.parameters()]

    @property
    def buffers(self):
        return [buffer for buffer in self._model.buffers()]

    @property
    def parameter_names(self):
        return [name for (name, _) in self._model.named_parameters()]

    def train_iterator(self, batch_size: int) -> Iterable[Batch]:
        """Create a dataloader serving `Batch`es from the training dataset.
        Example:
            >>> for batch in task.train_iterator(batch_size=32):
            ...     batch_loss, gradients = task.batchLossAndGradient(batch)
        """
        if batch_size == 10:
            # Define the fixed indices
            if self.ds == 'cifar100':
                fixed_indices = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
            elif self.ds == 'cifar10':
                fixed_indices = [201, 202, 203, 204, 212, 214, 216, 217, 219, 248]
            elif self.ds == 'mnist':
                fixed_indices = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
            fixed_indices = [i - 1 for i in fixed_indices]

            # Create a subset for the first fixed batch
            fixed_batch_subset = Subset(self._train_set, fixed_indices)

            # Create a subset for the remaining data
            remaining_indices = [i for i in range(len(self._train_set)) if i not in fixed_indices]
            remaining_dataset = Subset(self._train_set, remaining_indices)

            # Combine the fixed batch subset and the remaining dataset
            combined_dataset = ConcatDataset([fixed_batch_subset, remaining_dataset])

            # Create a DataLoader for the combined dataset
            train_loader = DataLoader(
                combined_dataset,
                batch_size=batch_size,
                shuffle=False,  # No need to shuffle, as we want fixed order
                pin_memory=True,
                num_workers=2,
            )

        else:
            # For other batch sizes, use the standard method
            sampler = DistributedSampler(dataset=self._train_set, add_extra_samples=True)
            sampler.set_epoch(self._epoch)

            train_loader = DataLoader(
                self._train_set,
                batch_size=batch_size,
                sampler=sampler,
                pin_memory=True,
                drop_last=True,
                num_workers=2,
            )

        self._epoch += 1

        return BatchLoader(train_loader, self._device)

    def batch_loss(self, batch: Batch) -> (float, Dict[str, float]):
        """
        Evaluate the loss on a batch.
        If the model has batch normalization or dropout, this will run in training mode.
        Returns:
            - loss function (float)
            - bunch of metrics (dictionary)
        """
        with torch.no_grad():
            with self._timer("batch.forward", float(self._epoch)):
                prediction = self._model(batch.x)
                loss = self._criterion(prediction, batch.y)
            with self._timer("batch.evaluate", float(self._epoch)):
                metrics = self.evaluate_prediction(prediction, batch.y)
        return loss.item(), metrics

    def batch_loss_and_gradient(
            self, batch: Batch
    ) -> (float, List[torch.Tensor], Dict[str, float]):
        """
        Evaluate the loss and its gradients on a batch.
        If the model has batch normalization or dropout, this will run in training mode.
        Returns:
            - function value (float)
            - gradients (list of tensors in the same order as task.state())
            - bunch of metrics (dictionary)
        """
        self._zero_grad()
        with self._timer("batch.forward", float(self._epoch)):
            prediction = self._model(batch.x)
            f = self._criterion(prediction, batch.y)
        with self._timer("batch.backward", float(self._epoch)):
            f.backward()
        with self._timer("batch.evaluate", float(self._epoch)):
            metrics = self.evaluate_prediction(prediction, batch.y)
        df = [parameter.grad for parameter in self._model.parameters()]
        return f.detach(), df, metrics

    def evaluate_prediction(self, model_output, reference):
        """
        Compute a series of scalar loss values for a predicted batch and references
        """
        with torch.no_grad():
            _, top5 = model_output.topk(5)
            top1 = top5[:, 0]
            cross_entropy = self._criterion(model_output, reference)
            accuracy = top1.eq(reference).sum().float() / len(reference)
            top5_accuracy = reference.unsqueeze(1).eq(top5).sum().float() / len(reference)
            return {
                "cross_entropy": cross_entropy.detach(),
                "accuracy": accuracy.detach(),
                "top5_accuracy": top5_accuracy.detach(),
            }

    def test(self, state_dict=None) -> float:
        """
        Compute the average loss on the test set.
        The task is completed as soon as the output is below self.target_test_loss.
        If the model has batch normalization or dropout, this will run in eval mode.
        """
        test_loader = BatchLoader(
            DataLoader(
                self._test_set,
                batch_size=250,
                num_workers=1,
                drop_last=False,
                pin_memory=True,
                sampler=DistributedSampler(dataset=self._test_set, add_extra_samples=False),
            ),
            self._device,
        )

        if state_dict:
            test_model = self._create_test_model(state_dict)
        else:
            test_model = self._model
            test_model.eval()

        mean_metrics = MeanAccumulator()

        for batch in test_loader:
            with torch.no_grad():
                prediction = test_model(batch.x)
                metrics = self.evaluate_prediction(prediction, batch.y)
            mean_metrics.add(metrics)
        mean_metrics.reduce()  # Collect over workers

        test_model.train()
        return mean_metrics.value()

    def state_dict(self):
        """Dictionary containing the model state (buffers + tensors)"""
        return self._model.state_dict()

    def _create_model(self):
        """Create a PyTorch module for the model"""
        torch.random.manual_seed(self._seed)
        model = getattr(cifar_architectures, self._architecture)(use_batchnorm = self.bn, ds = self.ds)
        model.to(self._device)
        model.train()
        return model

    def _create_test_model(self, state_dict):
        test_model = deepcopy(self._model)
        test_model.load_state_dict(state_dict)
        test_model.eval()
        return test_model

    def _create_dataset(self, data_root="./data"):
        """Create train and test datasets"""
        if self.ds == 'mnist':
            dataset = torchvision.datasets.MNIST

            data_mean = (0.1307,)  # Mean of MNIST dataset
            data_stddev = (0.3081,)  # Standard deviation of MNIST dataset

            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # Optional: Adds a slight augmentation
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_stddev)
            ])

            transform_test = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(data_mean, data_stddev),
                ]
            )

            training_set = dataset(root=data_root, train=True, download=True, transform=transform_train)
            test_set = dataset(root=data_root, train=False, download=True, transform=transform_test)

            return training_set, test_set

        if self.ds == 'cifar10':
            dataset = torchvision.datasets.CIFAR10
            data_mean = (0.4914, 0.4822, 0.4465)
            data_stddev = (0.2023, 0.1994, 0.2010)
        elif self.ds == 'cifar100':
            dataset = torchvision.datasets.CIFAR100
            data_mean = (0.5071, 0.4867, 0.4408)
            data_stddev = (0.2675, 0.2565, 0.2761)

        transform_train = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )

        transform_test = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(data_mean, data_stddev),
            ]
        )

        training_set = dataset(root=data_root, train=True, download=True, transform=transform_train)
        test_set = dataset(root=data_root, train=False, download=True, transform=transform_test)

        return training_set, test_set


    def _zero_grad(self):
        self._model.zero_grad()


class BatchLoader:
    """
    Utility that transforms a DataLoader that is an iterable over (x, y) tuples
    into an iterable over Batch() tuples, where its contents are already moved
    to the selected device.
    """

    def __init__(self, dataloader, device):
        self._dataloader = dataloader
        self._device = device

    def __len__(self):
        return len(self._dataloader)

    def __iter__(self):
        for x, y in self._dataloader:
            x = x.to(self._device)
            y = y.to(self._device)
            yield Batch(x, y)