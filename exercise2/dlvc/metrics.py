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
        self.classes = classes#
        self.num_classes = len(classes)

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.target_list  = []
        self.prediction_list = []
        self.needs_recalculation = True
        self.mIoU_val = 0

    def update(self, prediction: torch.Tensor,
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''
        prediction = prediction.detach().cpu()
        target = target.detach().cpu()

        # Check if prediction has the correct shape
        if prediction.shape[1] != len(self.classes):
            raise ValueError(f"Prediction tensors second dimension must match length of classes: {
                             len(self.classes)} ")

        # Check if target has the correct shape
        if prediction.shape[0] != target.shape[0] or prediction.shape[2] != target.shape[1] or prediction.shape[3] != target.shape[2]:
            raise ValueError(f"Prediction shape does not match target shape: {
                             prediction.shape} vs {target.shape} ")

        self.needs_recalculation = True

        prediction = torch.argmax(prediction, dim=1)
        prediction = torch.nn.functional.one_hot(prediction, num_classes=self.num_classes).movedim(-1, 1)

        indeces = target == 255
        target[indeces] = self.num_classes
        target = torch.nn.functional.one_hot(target, num_classes=20 if self.num_classes == 19 else 3).movedim(-1, 1) 

        if self.num_classes == 19:
            mask = target[:, -1, :, :] 
            target = target[:, :-1, :, :]
            prediction = self.masking(prediction, mask)

        self.target_list.append(target)
        self.prediction_list.append(prediction)

    def masking(self, input, mask):
        input = torch.permute(input, (1, 2, 3, 0))
        mask = torch.abs(torch.permute(mask, (1, 2, 0))-1)
        return torch.permute(input*mask, (3, 0, 1, 2))

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
        if not self.needs_recalculation:
            return self.mIoU_val
        
        if len(self.target_list) != 0 and len(self.prediction_list) != 0:
            predictions = torch.cat(self.prediction_list, dim=0)
            targets = torch.cat(self.target_list, dim=0)

            dim = [2, 3]

            intersection = torch.sum(predictions & targets, dim=dim)
            union =  torch.sum(targets, dim=dim) + torch.sum(predictions, dim=dim) - intersection        

            IoU = (intersection + 1e-7) / (union + 1e-7)
            mIoU = torch.mean(IoU).item()
        else:
            mIoU = float(0)

        self.mIoU_val = mIoU
        self.needs_recalculation = False
        return mIoU
