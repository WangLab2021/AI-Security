# Transferable Targeted Adversarial Attack(TTAA)

This repository contains the code for the paper:

[Towards Transferable Targeted Adversarial Examples](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Towards_Transferable_Targeted_Adversarial_Examples_CVPR_2023_paper.html) (CVPR 2023)



## Requirements

*   Python 3.7

*   torch 1.12.1

*   torchvision 0.13.1

*   numpy 1.21.6

## Experiments

#### Introduction

*   `attack.py` : the code for training the generator and discriminator on different models.

*   `generator.py` : The Network Architecture of generator.

*   `label_discriminator.py` : The Network Architecture of label discriminator.

*   `feature_discriminator.py` : The Network Architecture of feature discriminator.

#### Example Usage

    python train.py --src_dir dataset/source --match_dir dataset/target --feature_layer 5 --model_type Resnet18 --batch_size 64 --save_dir ./saved_model

## Citing this work

If you find this work is useful in your research, please consider citing:

    @inproceedings{wang2023towards,
      title={Towards transferable targeted adversarial examples},
      author={Wang, Zhibo and Yang, Hongshan and Feng, Yunhe and Sun, Peng and Guo, Hengchang and Zhang, Zhifei and Ren, Kui},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={20534--20543},
      year={2023}
    }

