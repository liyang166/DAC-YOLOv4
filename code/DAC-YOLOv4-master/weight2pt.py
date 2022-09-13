import torch
#from models import Darknet, load_darknet_weights
from models import Darknet
import numpy as np
import os

def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'
    # cutoff: save layers between 0 and cutoff (if cutoff = -1 all are saved)
    weights_file = weights.split(os.sep)[-1]

    # Try to download weights if not available locally
#    if not os.path.isfile(weights):
#        try:
#            os.system('wget https://pjreddie.com/media/files/' + weights_file + ' -O ' + weights)
#        except IOError:
#            print(weights + ' not found.\nTry https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI')

    # Establish cutoffs
    if weights_file == 'darknet53.conv.74':
        cutoff = 75
    elif weights_file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Open the weights file
    with open(weights, 'rb') as f:
        header = np.fromfile(f, dtype=np.int32, count=4)  #yolov3是4 yolov3-spp以上是5 First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]  # number of images seen during training
        weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

    ptr = 0
    for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if module_def['type'] == 'convolutional':
            conv_layer = module[0]
            if module_def['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn_layer = module[1]
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
            else:
                # Load conv. bias
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w

    return cutoff

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cfg = '/home/yy/code/C++/darknet-master-summer/strawberry_disease/sci/yolov4_my_backbone1_CBM_neck1_cbam_kmeans/strawberry_disease_type_3_yolov4_my_best.cfg'
weights = '/home/yy/code/C++/darknet-master-summer/strawberry_disease/sci/yolov4_my_backbone1_CBM_neck1_cbam_kmeans/backup_best/strawberry_disease_type_3_yolov4_my_best.weights'

model = Darknet(cfg).to(device)
load_darknet_weights(model, weights)
chkpt = {'epoch': -1, 'best_loss': None, 'model': model.state_dict(), 'optimizer': None}
torch.save(chkpt, 'strawberry_disease_type_3_yolov4_my_best.pt')
