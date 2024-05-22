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


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.target_list = []
        self.prediction_list = []

    def update(self, prediction: torch.Tensor,
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''
        prediction = prediction.cpu()
        target = target.cpu()

        # Check if prediction has the correct shape
        if prediction.shape[1] != len(self.classes):
            raise ValueError(f"Prediction tensors second dimension must match length of classes: {
                             len(self.classes)} ")

        # Check if target has the correct shape
        if prediction.shape[0] != target.shape[0] or prediction.shape[2] != target.shape[1] or prediction.shape[3] != target.shape[2]:
            raise ValueError(f"Prediction shape does not match target shape: {
                             prediction.shape} vs {target.shape} ")

        # # Check if the number of classes in prediction matches the number of classes in target
        # if np.max(np.array(target)) > len(self.classes)-1 or np.min(np.array(target)) < 0:
        #     raise ValueError(f"target array contains false values")

        prediction_argmax = np.array(np.argmax(prediction, axis=1))
        self.target_list += list(np.array(target))
        self.prediction_list += list(prediction_argmax)

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIoU: {round(self.mIoU(), 4)}\n"

    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        if len(self.target_list) != 0 and len(self.prediction_list) != 0:

            target = np.array(self.target_list)
            pred = np.array(self.prediction_list)
            _shape = target.shape

            # tp = true positive = A and B
            tp = np.zeros(len(self.classes))
            # ac = all classifications = A + B
            ac = np.zeros(len(self.classes))
            for i in range(_shape[0]):
                for j in range(_shape[1]):
                    for k in range(_shape[2]):
                        if pred[i][j][k] == target[i][j][k]:
                            tp[target[i][j][k]] += 1
                        ac[pred[i][j][k]] += 1
                        ac[target[i][j][k]] += 1
            mIoU = 0
            for i in range(self.classes):
                if ac[i] != 0:
                    mIoU += tp[i]/(ac[i]-tp[i])
            else:
                mIoU += 1
            mIoU = mIoU/self.classes

        else:
            mIoU = float(0)

        return mIoU
