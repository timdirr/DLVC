from abc import ABCMeta, abstractmethod
import torch
from dlvc.metrics import  Accuracy       
import torch
import numpy as np




    
#%%    


classes = ["a","b","c","d"]
    
a = Accuracy(classes)

print(a.accuracy())
print(a.per_class_accuracy())
print(a.per_class_accuracy())
print(a)



#%%

def random_arrays_with_indices(shape=(4, 4)):
    # Generate random numbers
    random_data = np.random.rand(*shape)
    
    # Normalize each row to sum up to 1
    normalized_data = random_data / np.sum(random_data, axis=1, keepdims=True)
    
    # Generate random indices between 0 and 3
    random_indices = np.random.randint(0, 4, size=(shape[0],))
    
    return normalized_data, random_indices



for i in range(20):
    
    
    prediction_np, target_np =  random_arrays_with_indices()
    
    
    prediction          = torch.tensor(prediction_np)
    target              = torch.tensor(target_np)  
    
    a.update(prediction,target)
    
    print(a)
    

prediction_np   = np.array([[0.5, 0.2, 0.3, 0.0],
                        [0.1, 0.1, 0.7, 0.1],
                        [0.1, 0.5, 0.2, 0.2],
                        [0.05, 0.05, 0.1, 0.8]])


target_np       = np.array([0,
                            1,
                            1,
                            3])


prediction          = torch.tensor(prediction_np)
target              = torch.tensor(target_np) 

a.reset()


a.update(prediction,target)
print(a)



#%%


print("Now checking Errors")


# prediction col missmatch
prediction_np   = np.array([[0.5, 0.2, 0.3],
                        [0.1, 0.1, 0.7],
                        [0.1, 0.5, 0.2],
                        [0.05, 0.05, 0.1]])

target_np       = np.array([0,
                            1,
                            1,
                            3])

prediction          = torch.tensor(prediction_np)
target              = torch.tensor(target_np) 

a.update(prediction,target)

#%%       

# target size missmatch

prediction_np   = np.array([[0.5, 0.2, 0.3, 0.0],
                        [0.1, 0.1, 0.7, 0.1],
                        [0.1, 0.5, 0.2, 0.2],
                        [0.05, 0.05, 0.1, 0.8]])


target_np       = np.array([0,
                            1,
                            1])


prediction          = torch.tensor(prediction_np)
target              = torch.tensor(target_np)

a.update(prediction,target)  


#%%

# value missmatch

prediction_np   = np.array([[0.5, 0.2, 0.3, 0.0],
                        [0.1, 0.1, 0.7, 0.1],
                        [0.1, 0.5, 0.2, 0.2],
                        [0.05, 0.05, 0.1, 0.8]])


target_np       = np.array([0,
                            1,
                            1,
                            5])


prediction          = torch.tensor(prediction_np)
target              = torch.tensor(target_np)      

a.update(prediction,target) 