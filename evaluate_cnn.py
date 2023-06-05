import os
import torch
from torch.autograd import Variable as V
from torch import nn
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from torch.utils.data import DataLoader
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
from dataset import CNNDataset

batch_size = 10


adv_dir = './advimages/model_vit_base_patch16_224-method_TGR'


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval().cuda(),)
    return model

def verify(model_name, path):

    model = get_model(model_name, path)

    dataset = CNNDataset("inc-v3", adv_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sum = 0
    for batch_idx, batch_data in enumerate(data_loader):
        batch_x = batch_data[0].cuda()
        batch_y = batch_data[1].cuda()
        batch_name = batch_data[2]

        with torch.no_grad():
            sum += (model(batch_x)[0].argmax(1) != batch_y+1).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / 1000.0))

def main():
    model_names = ['tf2torch_inception_v3','tf2torch_inception_v4','tf2torch_inc_res_v2','tf2torch_resnet_v2_101','tf2torch_ens3_adv_inc_v3','tf2torch_ens4_adv_inc_v3','tf2torch_ens_adv_inc_res_v2']

    models_path = './models/'
    for model_name in model_names:
        verify(model_name, models_path)
        print("===================================================")

if __name__ == '__main__':
    main()