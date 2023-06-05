import argparse
import os
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from utils import ROOT_PATH
import torch
from torch import nn
from Normalize import Normalize, TfNormalize
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

MODEL_NAMES = ['vit_base_patch16_224', 
               'deit_base_distilled_patch16_224', 
               'levit_256', 
               'pit_b_224', 
               'cait_s24_224', 
               'convit_base', 
               'tnt_s_patch16_224', 
               'visformer_small']

CORR_CKPTS = ['jx_vit_base_p16_224-4ee7a4dc.pth',
              'deit_base_distilled_patch16_224-df68dfff.pth',
              'LeViT-256-13b5763e.pth',
              'pit_b_820.pth',
              'S24_224.pth',
              'convit_base.pth',
              'tnt_s_patch16_224.pth.tar',
              'visformer_small-839e1f5b.pth']

def get_model(model_name):
        if model_name in MODEL_NAMES:
                model = create_model(
                        model_name,
                        pretrained=True,
                        num_classes=1000,
                        in_chans=3,
                        global_pool=None,
                        scriptable=False)  
        print ('Loading Model.')
        return model