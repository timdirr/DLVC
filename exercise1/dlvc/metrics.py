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
        
        ###### implemented code ########
        
        self.target_list        = []
        self.prediction_list    = [] 
        self.current_acc        = 0
        self.current_class_acc  = np.zeros(len(self.classes)) 
        self.curr_per_class_acc = 0
         
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        ## TODO implement
        
        ###### implemented code ########
        
        self.target_list        = []
        self.prediction_list    = []
        self.current_acc        = 0
        self.current_class_acc  = np.zeros(len(self.classes))
        self.curr_per_class_acc = 0
        
        pass

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''
        
        #TODO: wird hier danach abgefragt ob die Summe 1 ergibt? with each row being a class-score vector
        
        # Check if prediction has the correct shape
        if prediction.shape[1]!= len(self.classes):
            raise ValueError(f"Prediction tensors second dimension must match length of classes: {len(self.classes)} ")
    
        # Check if target has the correct shape
        if prediction.shape[0] != target.shape[0]:
            raise ValueError(f"Prediction tensors second dimension: {prediction.shape[0]} must the number of targets: {target.shape[0]}.")
    
        # Check if the number of classes in prediction matches the number of classes in target
        if np.max(np.array(target))>len(self.classes)-1 or np.min(np.array(target)) < 0:
            raise ValueError(f"target array contains false values")
            
        prediction_argmax        = np.array(np.argmax(prediction.detach().numpy(),-1))
        self.target_list         += list(np.array(target))
        self.prediction_list     += list(prediction_argmax)
        # TODO errors 
        
        self.accuracy()
        self.per_class_accuracy()
        
        
        ## TODO implement
        #pass
        
        return None # muss das?

    def __str__(self):
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''
        

            

        ## TODO implement
        
        ##### implemented code #####
        
        text = ""
        text +=f"accuracy: {self.current_acc}"
        text +="\n"
        text +=f"per class accuracy: {self.curr_per_class_acc}"
        text +="\n"
        for i,c in enumerate(self.classes):
            text +=f"Accuracy for class {self.classes[i]}: is {self.current_class_acc[i]}"
            text +="\n"
        

        return text


    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        
        #### implemented code ####
        if len(self.target_list) != 0 and len(self.prediction_list) != 0:
        
            target_arr          = np.array(self.target_list)
            prediction_arr      = np.array(self.prediction_list)
        
            overlap             = target_arr == prediction_arr        
            self.current_acc    = np.mean(overlap)
        
        else:
            self.current_acc    = float(0) 
            
        
        return self.current_acc
        
        
        

        ## TODO implement
        #pass
    
    def per_class_accuracy(self) -> float:
        '''
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        ## TODO implement
        
        
        #### implemented code ####
        target_arr          = np.array(self.target_list)
        prediction_arr      = np.array(self.prediction_list)
        
        
        class_based_scores = np.zeros(len(self.classes))

        for i,c in enumerate(self.classes):
            
            relevant_predictions = prediction_arr[target_arr==i]
            
            class_based_scores[i] = np.mean(relevant_predictions==i)
         
        class_based_scores[np.isnan(class_based_scores)] = 0  
        
        self.current_class_acc = class_based_scores
        self.curr_per_class_acc = np.mean(class_based_scores)
        
        
        return self.curr_per_class_acc
       