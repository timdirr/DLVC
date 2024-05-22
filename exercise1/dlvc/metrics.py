from abc import ABCMeta, abstractmethod
import torch
import numpy as np


class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.target_list = []
        self.prediction_list = []
        self.current_class_acc = np.zeros(len(self.classes))

    def update(self, prediction: torch.Tensor,
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''
        prediction = prediction.cpu()
        target = target.cpu()

        # Check if prediction has the correct shape
        if prediction.shape[1] != len(self.classes):
            raise ValueError(f"Prediction tensors second dimension must match length of classes: {
                             len(self.classes)} ")

        # Check if target has the correct shape
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(f"Prediction tensors second dimension: {
                             prediction.shape[0]} must the number of targets: {target.shape[0]}.")

        # Check if the number of classes in prediction matches the number of classes in target
        if np.max(np.array(target)) > len(self.classes)-1 or np.min(np.array(target)) < 0:
            raise ValueError(f"target array contains false values")

        prediction_argmax = np.array(
            np.argmax(prediction.detach().numpy(), -1))
        self.target_list += list(np.array(target))
        self.prediction_list += list(prediction_argmax)

    def __str__(self) -> str:
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''
        current_acc = self.accuracy()
        curr_per_class_acc = self.per_class_accuracy()

        text = f"accuracy: {round(current_acc, 4)}\n"
        text += f"per class accuracy: {round(curr_per_class_acc, 4)}\n"
        for i, c in enumerate(self.classes):
            text += f"Accuracy for class: {self.classes[i]}   \tis {
                round(self.current_class_acc[i], 2)}\n"

        return text

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        if len(self.target_list) != 0 and len(self.prediction_list) != 0:

            target_arr = np.array(self.target_list)
            prediction_arr = np.array(self.prediction_list)

            overlap = target_arr == prediction_arr
            current_acc = np.mean(overlap)

        else:
            current_acc = float(0)

        return current_acc

    def per_class_accuracy(self) -> float:
        '''
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        target_arr = np.array(self.target_list)
        prediction_arr = np.array(self.prediction_list)

        class_based_scores = np.zeros(len(self.classes))

        for i, c in enumerate(self.classes):

            relevant_predictions = prediction_arr[target_arr == i]

            class_based_scores[i] = np.mean(relevant_predictions == i)

        class_based_scores[np.isnan(class_based_scores)] = 0

        self.current_class_acc = class_based_scores

        return np.mean(class_based_scores)
