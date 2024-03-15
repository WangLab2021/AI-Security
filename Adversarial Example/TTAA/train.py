import argparse
import os
import datetime
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from feature_discriminator import FeatureDiscriminator
from label_discriminator import *
from torchvision.models import vgg16, resnet50, vgg19_bn, densenet121
from generators import *
from gaussian_smoothing import *

parser = argparse.ArgumentParser(description='Transferable Targeted Perturbations')
parser.add_argument('--src_dir', default='data', help='Path to data folder with source domain samples')
parser.add_argument('--match_dir', default='data/target', help='Path to data folder with target domain samples')
parser.add_argument('--match_target', type=int, default=24, help='target class(of ImageNet)')
parser.add_argument('--feature_layer', type=int, default=5, help='Extract feature of the label discriminator')
parser.add_argument('--batch_size', type=int, default=10, help='Number of training samples/batch')
parser.add_argument('--epochs', type=int, default=64, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate for adam')
parser.add_argument('--eps', type=int, default=16, help='Perturbation Budget during training, eps')
parser.add_argument('--model_type', type=str, default='Vgg16', help='Model under attack (discrimnator)')
parser.add_argument('--save_dir', type=str, default='save_model', help='Directory to save generators and AuxNet')
args = parser.parse_args()
print(args)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

eps = args.eps / 255

# GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Input dimensions
if args.model_type == 'inception_v3':
    scale_size = 300
    img_size = 299
else:
    scale_size = 256
    img_size = 224

# Data
transform = transforms.Compose([
    transforms.Resize(scale_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor()])

train_set = torchvision.datasets.ImageFolder(
    root=os.path.join(args.src_dir, 'train'),
    transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False)
test_set = torchvision.datasets.ImageFolder(
    root=os.path.join(args.src_dir, 'test'),
    transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=False)

target_set = torchvision.datasets.ImageFolder(
    root=args.match_dir,
    transform=transform)
target_loader = torch.utils.data.DataLoader(
    target_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False)
dataiter = iter(target_loader)

# Generator
if args.model_type == 'inception_v3':
    netG = GeneratorResnet(inception=True)
else:
    netG = GeneratorResnet()
netG.to(device)

# Feature Discriminator
f_d = FeatureDiscriminator(128)
f_d.to(device)

# Label_Discriminator & Feature_Extractor
l_d = eval(args.model_type)()
l_d = l_d.to(device)



# Target model
t_model = resnet50(pretrained=True).to(device).eval()

# Optimizer
optimG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimD = optim.Adam(f_d.parameters(), lr=args.lr, betas=(0.5, 0.999))

BCE = nn.BCELoss().to(device)
CE = nn.CrossEntropyLoss().to(device)
KL = nn.KLDivLoss().to(device)

# ----------
#  Training
# ----------
for epoch in range(args.epochs):
    D_batch_real_loss = 0
    D_batch_fake_loss = 0
    G_batch_class_loss = 0
    G_batch_feature_loss = 0
    for i, (img, _) in enumerate(train_loader):
        netG.train()
        f_d.train()

        img = img.to(device)
        try:
            t_img = next(dataiter)[0]
        except StopIteration:
            dataiter = iter(target_loader)
            t_img = next(dataiter)[0]
        t_img = t_img.to(device)
        
        _, t_features = l_d(normalize(t_img.clone().detach()))
        t_feature = t_features[feature_layer]
        t_feature_similarity = f_d(t_feature)

        adv = netG(img.clone())

        # Projection
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = torch.clamp(adv, 0.0, 1.0)
        
        _, s_features = l_d(normalize(adv.clone().detach()))
        s_feature = s_features[feature_layer]
        s_feature_similarity = f_d(feature)

        # Update the feature discriminator
        D_fake_loss = BCE(s_feature_similarity, torch.zeros(outputs.shape).detach().to(device))
        D_real_loss = BCE(t_feature_similarity, torch.ones(t_outputs.shape).detach().to(device))
        D_loss = 0.5 * D_fake_loss + 0.5 * D_real_loss

        optimD.zero_grad()
        D_loss.backward(retain_graph=True)
        optimD.step()
               
        # Update the generator
        s_pred, s_features = l_d(normalize(adv.clone().detach()))
        s_feature = s_features[feature_layer]
        s_feature_similarity = f_d(s_feature)
        t_pred, _ = l_d(normalize(t_img.clone().detach()))

        G_feature_loss = BCE(s_feature_similarity, torch.ones(outputs.shape).detach().to(device))
        G_class_loss = KL(F.log_softmax(s_pred, dim=1), t_pred) + KL(F.log_softmax(t_pred, dim=1), s_pred)
        G_loss = G_feature_loss + G_class_loss  

        optimG.zero_grad()       
        G_loss.backward()
        optimG.step()

        D_batch_real_loss += D_real_loss.item()
        D_batch_fake_loss += D_fake_loss.item()
        G_batch_class_loss += G_class_loss.item()
        G_batch_feature_loss += G_feature_loss.item()

        if (i + 1) % 500 == 0:
            print('Epoch: {0}/{1} \t Batch: {2}/{3} \t Generator class loss: {4:.3f} \t feature loss: {5:.3f} \t Discriminator real loss: {6:.3f} \t fake loss: {7:.3f}'\
                  .format(epoch, args.epochs, i + 1, len(train_loader), G_batch_class_loss, G_batch_feature_loss, D_batch_real_loss, D_batch_fake_loss))
            G_batch_class_loss = 0
            G_batch_feature_loss = 0
            D_batch_real_loss = 0
            D_batch_fake_loss = 0

    with torch.no_grad():
        target_fooling_rate = 0
        for i, (img, label) in enumerate(test_loader):
            img = img.to(device)
            label = label.to(device)

            adv = netG(img.clone()) 
            # Projection
            adv = torch.min(torch.max(adv, img - eps), img + eps)
            adv = torch.clamp(adv, 0.0, 1.0)


            outputs = t_model(normalize(adv)) 
            pred = outputs.argmax(dim=1) 

            target_fooling_rate += torch.sum(pred == args.match_target)
            # res.extend(list(pred.cpu().numpy()))
        print(
            '【Epoch: {0}/{1}】\t【TARGET Transfer Fooling Rate: {2}】'.format(
                epoch, args.epochs,
                target_fooling_rate / len(test_set)))
        # cnt = Counter(res)
        # print('val:', cnt)

    torch.save(netG.state_dict(),
               args.save_dir + '/netG_{}_{}_{}.pth'.format(args.model_type, epoch, args.match_target))
    torch.save(f_d, args.save_dir + '/f_d_{}_{}_{}.pth'.format(args.model_type, epoch, args.match_target))

