# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:18:22 2024

@author: ericx
"""

import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import glob
from matplotlib import pyplot as plt

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        
        dictt = {}
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]

        
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
            dictt[tag]  = r
            
            
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data,dictt


plt.close('all')

ROOT_DIR = os.getcwd()

DIR = os.path.join(ROOT_DIR,"training","fcn_resnet50")

subfolders      = [ f.path for f in os.scandir(DIR) if f.is_dir() ]
files           = []
subfolder_names = [] 
for f in subfolders:
    file = glob.glob(os.path.join(f,"events*"))[0]
    files.append(file)
    subfolder_names.append(os.path.basename(f))


# Experiment 1
# =============================================================================
# relevant_runs = [
#     "initial_config",
#     "initial_config_lr_fixed_0002",
#     "aug_hflip_crop_kaggle_ref_100_epoch_lr_0_002",
#     "dropout_20_ref_100_epoch_lr_0_002",
#     "dropout_20_aug_hflip_crop_kaggle_ref_100_epoch_lr_0_002",
#     "dropout_20_weight_decay_1_aug_hflip_crop_kaggle_ref_100_epoch_lr_0_002",
#     "modified_resnet18",
#     "modified_resnet18_adam_cosineannealinglr",
#     "modified_resnet18_adam_cosineannealinglr_gradclipping_dropout0.2_lr0.002"
#     ]
# 
# run_name = [
#     "M1: initial configuration",
#     "M2: learning rate = 0.002",
#     "M3: hflip, random crop",
#     "M4: dropout20",
#     "M5: hflip, random crop + dropout20",
#     "M6: hflip, random crop + dropout20 + weight decay1",
#     "M7: modified ResNet18 + SGD + Onecycle",
#     "M8: modified ResNet18 + Adam + cosine Annealing LR",
#     "M9: modified ResNet18 + Adam + cosine Annealing LR + gradclipping"
#     ]
# =============================================================================

# Experiment 2

relevant_runs = [
    "adamw_lr_0.0001_exponential_gamma_0.98_ep_30_wd_0.1",
    "pre_adamw_lr_0.0001_exponential_gamma_0.98_ep_30_wd_0.1",
    ]

run_name = [
    "F1: FCN from scratch",
    "F2: FCN pretrained"
    ]


dictt_list = []

for i,r in enumerate(relevant_runs):
    index = subfolder_names.index(r)
    
    print(files[index])
    
    aa,dictt=tflog2pandas(files[index])
    dictt_list.append(dictt)
  
    
# manipulations

# =============================================================================
# # decrease the number of epochs for dropout to 60 to increase visibility
# dictt_list[3]["Accuracy/train"] = dictt_list[3]["Accuracy/train"][0:60]
# dictt_list[3]["Accuracy/val"] = dictt_list[3]["Accuracy/val"][0:12]
# dictt_list[3]["PerClassAcc/train"] = dictt_list[3]["PerClassAcc/train"][0:60]
# dictt_list[3]["PerClassAcc/val"] = dictt_list[3]["PerClassAcc/val"][0:12]
# =============================================================================
    
#%%

# plt.close('all')


# Create a 2 by 2 subplot figure
fig, axs = plt.subplots(2, 2, figsize=(16, 8))

for i,d in enumerate(dictt_list):
    
    # Plot training accuracy
    axs[0, 0].plot(d["mIoU/train"]["step"],d["mIoU/train"]["value"],label = run_name[i])
    axs[0, 0].set_title('train/mIoU')
    axs[0, 0].set_ylim([0.3, 0.8]) 

    # Plot validation accuracy
    axs[0, 1].plot(d["mIoU/val"]["step"],d["mIoU/val"]["value"],label = run_name[i])
    axs[0, 1].set_title('val/mIoU')
    axs[0, 1].set_ylim([0.3, 0.8])

    # Plot mean per class training accuracy
    axs[1, 0].plot(d["Loss/train"]["step"],d["Loss/train"]["value"],label = run_name[i])
    axs[1, 0].set_title('train/Loss')
    axs[1, 0].set_ylim([0.0, 1.0])

    # Plot mean per class validation accuracy
    axs[1, 1].plot(d["Loss/val"]["step"],d["Loss/val"]["value"],label = run_name[i])
    axs[1, 1].set_title('val/Loss')
    axs[1, 1].set_ylim([0.0, 1.0])


handles, labels = axs[1, 1].get_legend_handles_labels()

# Create a single legend for all subplots and place it to the right with extra space
fig.legend(handles, labels, loc='upper center',fontsize="large")

# Adjust layout
plt.tight_layout(rect=[0, 0, 0, 0.80])
# plt.tight_layout()

# Show plot
plt.show()

# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True)


# =============================================================================
# 
# # Plot per class training accuracy
# for i in range(len(per_class_training_accuracy)):
#     axs[1, 0].plot(per_class_training_accuracy[i], label=f'Class {i}')
# axs[1, 0].set_title('Per Class Training Accuracy')
# axs[1, 0].legend()
# 
# # Plot per class validation accuracy
# for i in range(len(per_class_validation_accuracy)):
#     axs[1, 1].plot(per_class_validation_accuracy[i], label=f'Class {i}')
# axs[1, 1].set_title('Per Class Validation Accuracy')
# axs[1, 1].legend()
# 
# # Adjust layout
# plt.tight_layout()
# 
# # Show plot
# plt.show()
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# import matplotlib.pyplot as plt
# 
# # Example data (replace this with your actual data)
# training_accuracy = [0.8, 0.85, 0.9, 0.92]
# validation_accuracy = [0.75, 0.78, 0.82, 0.85]
# per_class_training_accuracy = [[0.75, 0.85, 0.82], [0.78, 0.88, 0.85], [0.8, 0.9, 0.87], [0.82, 0.92, 0.9]]
# per_class_validation_accuracy = [[0.7, 0.8, 0.78], [0.72, 0.82, 0.8], [0.75, 0.85, 0.82], [0.78, 0.88, 0.85]]
# 
# 
# 
# 
# 
# 
# # Create a 2 by 2 subplot figure
# fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# 
# # Plot training accuracy
# axs[0, 0].plot(training_accuracy)
# axs[0, 0].set_title('Training Accuracy')
# 
# # Plot validation accuracy
# axs[0, 1].plot(validation_accuracy)
# axs[0, 1].set_title('Validation Accuracy')
# 
# # Plot per class training accuracy
# for i in range(len(per_class_training_accuracy)):
#     axs[1, 0].plot(per_class_training_accuracy[i], label=f'Class {i}')
# axs[1, 0].set_title('Per Class Training Accuracy')
# axs[1, 0].legend()
# 
# # Plot per class validation accuracy
# for i in range(len(per_class_validation_accuracy)):
#     axs[1, 1].plot(per_class_validation_accuracy[i], label=f'Class {i}')
# axs[1, 1].set_title('Per Class Validation Accuracy')
# axs[1, 1].legend()
# 
# # Adjust layout
# plt.tight_layout()
# 
# # Show plot
# plt.show()
# 
# 
# 
# 
# =============================================================================



#path = os.path.join(ROOT_DIR,"tested_configs","resnet18","initial_config","events.out.tfevents.1714311836.DESKTOP-EG2I5L1.8492.0")











# path="Run1" #folderpath
#df,dictt=tflog2pandas(path)