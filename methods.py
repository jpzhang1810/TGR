import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
import scipy.stats as st
import copy
from utils import ROOT_PATH
from functools import partial
import copy
import pickle as pkl
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import params
from model import get_model

class BaseAttack(object):
    def __init__(self, attack_name, model_name, target):
        self.attack_name = attack_name
        self.model_name = model_name
        self.target = target
        if self.target:
            self.loss_flag = -1
        else:
            self.loss_flag = 1
        self.used_params = params(self.model_name)

        # loading model
        self.model = get_model(self.model_name)
        self.model.cuda()
        self.model.eval()

    def forward(self, *input):
        """
        Rewrite
        """
        raise NotImplementedError

    def _mul_std_add_mean(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        inps.mul_(std[:,None, None]).add_(mean[:,None,None])
        return inps

    def _sub_mean_div_std(self, inps):
        dtype = inps.dtype
        mean = torch.as_tensor(self.used_params['mean'], dtype=dtype).cuda()
        std = torch.as_tensor(self.used_params['std'], dtype=dtype).cuda()
        #inps.sub_(mean[:,None,None]).div_(std[:,None,None])
        inps = (inps - mean[:,None,None])/std[:,None,None]
        return inps

    def _save_images(self, inps, filenames, output_dir):
        unnorm_inps = self._mul_std_add_mean(inps)
        for i,filename in enumerate(filenames):
            save_path = os.path.join(output_dir, filename)
            image = unnorm_inps[i].permute([1,2,0]) # c,h,w to h,w,c
            image[image<0] = 0
            image[image>1] = 1
            image = Image.fromarray((image.detach().cpu().numpy()*255).astype(np.uint8))
            # print ('Saving to ', save_path)
            image.save(save_path)

    def _update_inps(self, inps, grad, step_size):
        unnorm_inps = self._mul_std_add_mean(inps.clone().detach())
        unnorm_inps = unnorm_inps + step_size * grad.sign()
        unnorm_inps = torch.clamp(unnorm_inps, min=0, max=1).detach()
        adv_inps = self._sub_mean_div_std(unnorm_inps)
        return adv_inps

    def _update_perts(self, perts, grad, step_size):
        perts = perts + step_size * grad.sign()
        perts = torch.clamp(perts, -self.epsilon, self.epsilon)
        return perts

    def _return_perts(self, clean_inps, inps):
        clean_unnorm = self._mul_std_add_mean(clean_inps.clone().detach())
        adv_unnorm = self._mul_std_add_mean(inps.clone().detach())
        return adv_unnorm - clean_unnorm

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images



class TGR(BaseAttack):
    def __init__(self, model_name, sample_num_batches=130, steps=10, epsilon=16/255, target=False, decay=1.0):
        super(TGR, self).__init__('TGR', model_name, target)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon/self.steps
        self.decay = decay

        self.image_size = 224
        self.crop_length = 16
        self.sample_num_batches = sample_num_batches
        self.max_num_batches = int((224/16)**2)
        assert self.sample_num_batches <= self.max_num_batches
        self._register_model()

    
    def _register_model(self):   
        def attn_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.model_name in ['vit_base_patch16_224', 'visformer_small', 'pit_b_224']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,:] = 0.0
                out_grad[:,range(C),:,max_all_W] = 0.0
                out_grad[:,range(C),min_all_H,:] = 0.0
                out_grad[:,range(C),:,min_all_W] = 0.0
                
            if self.model_name in ['cait_s24_224']:
                B,H,W,C = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B, H*W, C)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                min_all_H = min_all//H
                min_all_W = min_all%H
                
                out_grad[:,max_all_H,:,range(C)] = 0.0
                out_grad[:,:,max_all_W,range(C)] = 0.0
                out_grad[:,min_all_H,:,range(C)] = 0.0
                out_grad[:,:,min_all_W,range(C)] = 0.0

            return (out_grad, )
        
        def attn_cait_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            
            B,H,W,C = grad_in[0].shape
            out_grad_cpu = out_grad.data.clone().cpu().numpy()
            max_all = np.argmax(out_grad_cpu[0,:,0,:], axis = 0)
            min_all = np.argmin(out_grad_cpu[0,:,0,:], axis = 0)
                
            out_grad[:,max_all,:,range(C)] = 0.0
            out_grad[:,min_all,:,range(C)] = 0.0
            return (out_grad, )
            
        def q_tgr(module, grad_in, grad_out, gamma):
            # cait Q only uses class token
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            out_grad[:] = 0.0
            return (out_grad, grad_in[1], grad_in[2])
            
        def v_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            if self.model_name in ['visformer_small']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,max_all_W] = 0.0
                out_grad[:,range(C),min_all_H,min_all_W] = 0.0

            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                    
                out_grad[:,max_all,range(c)] = 0.0
                out_grad[:,min_all,range(c)] = 0.0
            return (out_grad, grad_in[1])
        
        def mlp_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            if self.model_name in ['visformer_small']:
                B,C,H,W = grad_in[0].shape
                out_grad_cpu = out_grad.data.clone().cpu().numpy().reshape(B,C,H*W)
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 1)
                max_all_H = max_all//H
                max_all_W = max_all%H
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 1)
                min_all_H = min_all//H
                min_all_W = min_all%H
                out_grad[:,range(C),max_all_H,max_all_W] = 0.0
                out_grad[:,range(C),min_all_H,min_all_W] = 0.0
            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'resnetv2_101']:
                c = grad_in[0].shape[2]
                out_grad_cpu = out_grad.data.clone().cpu().numpy()
        
                max_all = np.argmax(out_grad_cpu[0,:,:], axis = 0)
                min_all = np.argmin(out_grad_cpu[0,:,:], axis = 0)
                out_grad[:,max_all,range(c)] = 0.0
                out_grad[:,min_all,range(c)] = 0.0
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics
                

        attn_tgr_hook = partial(attn_tgr, gamma=0.25)
        attn_cait_tgr_hook = partial(attn_cait_tgr, gamma=0.25)
        v_tgr_hook = partial(v_tgr, gamma=0.75)
        q_tgr_hook = partial(q_tgr, gamma=0.75)
        
        mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)

        if self.model_name in ['vit_base_patch16_224' ,'deit_base_distilled_patch16_224']:
            for i in range(12):
                self.model.blocks[i].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.blocks[i].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.blocks[i].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.blocks[block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.blocks[block_ind].mlp.register_backward_hook(mlp_tgr_hook)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(attn_cait_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.q.register_backward_hook(q_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.k.register_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].attn.v.register_backward_hook(v_tgr_hook)
                    self.model.blocks_token_only[block_ind-24].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'visformer_small':
            for block_ind in range(8):
                if block_ind < 4:
                    self.model.stage2[block_ind].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.stage2[block_ind].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.stage2[block_ind].mlp.register_backward_hook(mlp_tgr_hook)
                elif block_ind >=4:
                    self.model.stage3[block_ind-4].attn.attn_drop.register_backward_hook(attn_tgr_hook)
                    self.model.stage3[block_ind-4].attn.qkv.register_backward_hook(v_tgr_hook)
                    self.model.stage3[block_ind-4].mlp.register_backward_hook(mlp_tgr_hook)

    def _generate_samples_for_interactions(self, perts, seed):
        add_noise_mask = torch.zeros_like(perts)
        grid_num_axis = int(self.image_size/self.crop_length)

        # Unrepeatable sampling
        ids = [i for i in range(self.max_num_batches)]
        random.seed(seed)
        random.shuffle(ids)
        ids = np.array(ids[:self.sample_num_batches])

        # Repeatable sampling
        # ids = np.random.randint(0, self.max_num_batches, size=self.sample_num_batches)
        rows, cols = ids // grid_num_axis, ids % grid_num_axis
        flag = 0
        for r, c in zip(rows, cols):
            add_noise_mask[:,:,r*self.crop_length:(r+1)*self.crop_length,c*self.crop_length:(c+1)*self.crop_length] = 1
        add_perturbation = perts * add_noise_mask
        return add_perturbation

    def forward(self, inps, labels):
        inps = inps.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(inps).cuda()
        unnorm_inps = self._mul_std_add_mean(inps)
        perts = torch.zeros_like(unnorm_inps).cuda()
        perts.requires_grad_()

        for i in range(self.steps):
            
            #add_perturbation = self._generate_samples_for_interactions(perts, i)
            #outputs = self.model((self._sub_mean_div_std(unnorm_inps + add_perturbation)))

            ##### If you use patch out, please uncomment the previous two lines and comment the next line.

            outputs = self.model((self._sub_mean_div_std(unnorm_inps + perts)))
            cost = self.loss_flag * loss(outputs, labels).cuda()
            cost.backward()
            grad = perts.grad.data
            grad = grad / torch.mean(torch.abs(grad), dim=[1,2,3], keepdim=True)
            grad += momentum*self.decay
            momentum = grad
            perts.data = self._update_perts(perts.data, grad, self.step_size)
            perts.data = torch.clamp(unnorm_inps.data + perts.data, 0.0, 1.0) - unnorm_inps.data
            perts.grad.data.zero_()
        return (self._sub_mean_div_std(unnorm_inps+perts.data)).detach(), None
        