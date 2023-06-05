# Token Gradient Regularization
Official Pytorch implementation for "Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization" (CVPR 2023).

**[Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization](https://arxiv.org/pdf/2303.15754.pdf)  (CVPR 2023)**

## Requirements

- Python 3.6.13
- Pytorch 1.7.1
- Torchvision 0.8.2
- Numpy 1.19.2
- Pillow 8.3.1
- Timm 0.4.12 
- Scipy 1.5.4

## Experiments


ViT models are all available in [timm](https://github.com/huggingface/pytorch-image-models) library. We consider four surrogate models (vit_base_patch16_224, pit_b_224, cait_s24_224, and visformer_small) and four additional target models (deit_base_distilled_patch16_224, levit_256, convit_base, tnt_s_patch16_224).

To evaluate CNN models, please download the converted pretrained models from ( https://github.com/ylhz/tf_to_pytorch_model) before running the code. Then place these model checkpoint files in `./models`.

#### Introduction


- `methods.py` : the implementation for TGR attack.

- `evaluate.py` : the code for evaluating generated adversarial examples on different ViT models.

- `evaluate_cnn.py` : the code for evaluating generated adversarial examples on different CNN models.
  

#### Example Usage

##### Generate adversarial examples:

- TGR

```
python attack.py --attack TGR --batch_size 1 --model_name vit_base_patch16_224
```

You can also modify the hyper parameter values to align with the detailed setting in our paper.


##### Evaluate the attack success rate

- Evaluate on ViT models

```
bash run_evaluate.sh model_vit_base_patch16_224-method_TGR
```

- Evaluate on CNN models

```
python evaluate_cnn.py
```


## Citing this work

If you find this work is useful in your research, please consider citing:

```
@inproceedings{zhang2023transferable,
  title={Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization},
  author={Zhang, Jianping and Huang, Yizhan and Wu, Weibin and Lyu, Michael R},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16415--16424},
  year={2023}
}
```

## Acknowledgments

Code refer to: [Towards Transferable Adversarial Attacks on Vision Transformers](https://github.com/zhipeng-wei/PNA-PatchOut) and [tf_to_torch_model](https://github.com/ylhz/tf_to_pytorch_model)
