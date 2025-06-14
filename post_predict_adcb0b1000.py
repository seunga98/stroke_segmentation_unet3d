import torch
import os
import numpy as np
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from model import MyUnet3d
from dataPreparation3 import AphasiaDWIData

USE_CUDA = True

""" Define DataSet """

PATH_DATASET = "/media/hdd_sda/seunga_seg_preproc_result_modify/"
PATH_CURR_RUN = "/media/hdd_sda/seg_seunga/./model_file"
PATH_EXPORT = "/media/hdd_sda/seunga_seg_preproc_result_export_modify/"

""" Load trained model parameters """

fn_model_params = os.path.join(PATH_CURR_RUN, 'model_shape.pth')
model_params = torch.load(fn_model_params)
BATCH_SIZE = 1
channel_in = model_params['channel_in']
channel_out = model_params['channel_out']
channel_filter = model_params['channel_filter']
MODALITY = model_params['MODALITY']
str_modals = " ".join(str(i) for i in MODALITY)

print('\t \t Model predicts')
print('\t- Input modality: %s'%str_modals)
print('\t- Unet Model [%d, %d, %d]'%(channel_in, channel_out, channel_filter))

dset = AphasiaDWIData(modality=MODALITY, path_dataset=PATH_DATASET)

fn_checkpoint = os.path.join(PATH_CURR_RUN, 'checkpoint.pth')
fn_checkloss_test = os.path.join(PATH_CURR_RUN, 'testset_loss.txt')
fn_checkloss = os.path.join(PATH_CURR_RUN, 'validset_loss.txt')
fn_checkbest = os.path.join(PATH_CURR_RUN, 'model_best.pth')

dset_loader = torch.utils.data.DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, pin_memory=True)

model = MyUnet3d(in_dim=channel_in, out_dim=channel_out, num_filter=channel_filter)

if USE_CUDA:
    model = nn.DataParallel(model)
    model = model.cuda()

""" -- Load previous models -- """
checkpoint = torch.load(fn_checkbest)
epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['model_state'])
print('Checkpoint, result at ', epoch, ' epoch')

""" Evaluation """
model.eval()
num_samples = dset_loader.dataset.__len__()

""" Using all the set """
t = time.time()
dataloader = dset_loader
num_samples = dataloader.__len__()
num_iteration_batch = np.int64(np.ceil(len(dset) / BATCH_SIZE))

for batch_idx, (data) in enumerate(dataloader):
    with torch.autograd.no_grad():
        if USE_CUDA:
            data = data.cuda()

        output = model(data)

        print('\t Batch [%03d / %03d]' % (
                  batch_idx + 1, num_iteration_batch), ', Elapsed %.2f' % (time.time() - t))
        
        if USE_CUDA:
            output = output.cpu()
        dataloader.dataset.export_image2(PATH_EXPORT, output, epoch, batch_idx)