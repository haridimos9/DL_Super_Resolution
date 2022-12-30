"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
import csv
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics import PeakSignalNoiseRatio
import neptune.new as neptune
from neptune.new.types import File
import torchvision.transforms as T
from PIL import Image

# Import writer class from csv module
from csv import writer
import logger

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--scale", type=int, default=4, help="zooming factor")
parser.add_argument("--dataset_name", type=str, default="splits/train", help="name of the training dataset")
parser.add_argument("--validation_name", type=str, default="splits/val", help="name of the validation dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample= int(int(opt.scale)/2)).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape, scale = opt.scale),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.validation_name, hr_shape=hr_shape, scale = opt.scale),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

#####

header = ['Epoch', 'Batch', 'D loss', 'G loss','content', 'adv', 'pixel' ]

with open('stats.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

run = neptune.init(
    project="DeepLearningDTU/ESRGAN",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYTIzMTU1ZC1hZDIyLTRkNzAtOTUyYS04MGRhODczYTJlOWYifQ==",
)  # your credentials

#params = {"learning_rate": opt.lr, "optimizer": "Adam"}
run["hyperparameters"] = opt
# ----------
#  Training
# ----------
def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[1:3]
    pred = pred[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    gt = gt[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    psnr = PeakSignalNoiseRatio().to(device)
    psnrprd = psnr (pred,gt)
    #imdff = pred - gt
    #print(imdff)
    #rmse = math.sqrt(np.mean(imdff ** 2))
    #if rmse == 0:
    #    return 100
    #return 20 * math.log10(255.0 / rmse)
    return psnrprd

def SSIM(pred, gt, shave_border=0):
    #height, width = pred.shape[2:4]
    #pred = pred[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    #gt = gt[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    # for i in range(opt.batch_size):
    #     print(pred.shape, gt.shape)
    #     ssimp[i] = structural_similarity_index_measure(torch.unsqueeze(pred[i,:,:,:], 0), torch.unsqueeze(gt[i,:,:,:], 0))
    #     ssimsum += ssimp[i]
    # ssimprd = ssimsum / opt.batch_size
    simprd = structural_similarity_index_measure(pred, gt)
    return ssimprd
#prev_ep = -1
for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        
        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        #####ADDED
        Simgs_hr = imgs_hr #nn.functional.interpolate(imgs_lr, scale_factor=4)
        with torch.no_grad():
            Sgen_hr = denormalize(gen_hr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
            )
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_content.item(),
                loss_GAN.item(),
                loss_pixel.item(),

            )
        )
        run["training_step/GLoss"].log(loss_G.item())
        run["training_step/DLoss"].log(loss_D.item())
        run["training_step/loss_content"].log(loss_content.item())
        run["training_step/loss_GAN"].log(loss_GAN.item())
        run["training_step/loss_pixel"].log(loss_pixel.item())

        #psnr_predicted = PSNR(Sgen_hr.to(device), Simgs_hr.type_as(Sgen_hr), shave_border=0)
        #print("PSNR Predicted = ", psnr_predicted)

        #ssim_predicted = SSIM(Sgen_hr.to(device), Simgs_hr.to(device), shave_border=0)
        #print("SSIM Predicted = ", ssim_predicted)

        #run["training_step/PSNR"].log(psnr_predicted)
        #run["training_step/SSIM"].log(ssim_predicted)
        vals = [str(epoch)+'/'+str(opt.n_epochs), str(i)+'/'+str(len(dataloader)), loss_D.item(), loss_G.item(), loss_content.item(), loss_GAN.item(), loss_pixel.item()]

        with open('stats.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(vals)

        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=int(opt.scale))
            img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
            #toimg = T.ToPILImage()
            #gridimg = make_grid(img_grid, nrow=1, normalize=False)
            #print(gridimg)
            #run["imgs/train_img"].upload(gridimg)
            save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)
            run["imgs/train_img"].upload("images/training/%d.png" % batches_done)

        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" %epoch)

    # for i, imgs in enumerate(val_dataloader):

        
    #     batches_done = epoch * len(val_dataloader) + i

    #     # Configure model input
    #     imgs_lr = Variable(imgs["lr"].type(Tensor))
    #     imgs_hr = Variable(imgs["hr"].type(Tensor))

    #     # Adversarial ground truths
    #     valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
    #     fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

    #     # ------------------
    #     #  Train Generators
    #     # ------------------

    #     optimizer_G.zero_grad()

    #     # Generate a high resolution image from low resolution input
    #     gen_hr = generator(imgs_lr)
        
    #     #####ADDED
    #     Simgs_hr = imgs_hr #nn.functional.interpolate(imgs_lr, scale_factor=4)
    #     Sgen_hr = gen_hr

    #     # Measure pixel-wise loss against ground truth
    #     loss_pixel = criterion_pixel(gen_hr, imgs_hr)


    #     # Extract validity predictions from discriminator
    #     pred_real = discriminator(imgs_hr).detach()
    #     pred_fake = discriminator(gen_hr)

    #     # Adversarial loss (relativistic average GAN)
    #     loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

    #     # Content loss
    #     gen_features = feature_extractor(gen_hr)
    #     real_features = feature_extractor(imgs_hr).detach()
    #     loss_content = criterion_content(gen_features, real_features)

    #     # Total generator loss
    #     loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

    #     # ---------------------
    #     #  Train Discriminator
    #     # ---------------------

    #     optimizer_D.zero_grad()

    #     pred_real = discriminator(imgs_hr)
    #     pred_fake = discriminator(gen_hr.detach())

    #     # Adversarial loss for real and fake images (relativistic average GAN)
    #     loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
    #     loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

    #     # Total loss
    #     loss_D = (loss_real + loss_fake) / 2

    #     # --------------
    #     #  Log Progress
    #     # --------------

    #     # def PSNR(pred, gt, shave_border=0):
    #     #     height, width = pred.shape[1:3]
    #     #     pred = pred[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    #     #     gt = gt[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    #     #     psnr = PeakSignalNoiseRatio().to(device)
    #     #     psnrprd = psnr (pred,gt)
    #     #     return psnrprd

    #     # def SSIM(pred, gt, shave_border=0):
    #     #     height, width = pred.shape[2:4]
    #     #     pred = pred[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    #     #     gt = gt[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    #     #     print(pred.shape, gt.shape)
    #     #     ssimprd = structural_similarity_index_measure(pred, gt)
    #     #     return ssimprd

        
    #     run["validation_step/GLoss"].log(loss_G.item())
    #     run["validation_step/DLoss"].log(loss_D.item())
    #     run["validation_step/loss_content"].log(loss_content.item())
    #     run["validation_step/loss_GAN"].log(loss_GAN.item())
    #     run["validation_step/loss_pixel"].log(loss_pixel.item())

    #     #psnr_predicted = PSNR(Sgen_hr.to(device), Simgs_hr.to(device), shave_border=0)
    #     #print("PSNR Predicted = ", psnr_predicted)

    #     #ssim_predicted = SSIM(Sgen_hr.to(device), Simgs_hr.to(device), shave_border=0)
    #     #print("SSIM Predicted = ", ssim_predicted)

    #     #run["validation_step/PSNR"].log(psnr_predicted)
    #     #run["validation_step/SSIM"].log(ssim_predicted)
    #     vals = [str(epoch)+'/'+str(opt.n_epochs), str(i)+'/'+str(len(val_dataloader)), loss_D.item(), loss_G.item(), loss_content.item(), loss_GAN.item(), loss_pixel.item()]

    #     with open('val_stats.csv', 'a', encoding='UTF8', newline='') as f:
    #         writer = csv.writer(f)

    #         # write the header
    #         writer.writerow(vals)

    #     if batches_done % opt.sample_interval == 0:
    #         # Save image grid with upsampled inputs and ESRGAN outputs
    #         imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=int(opt.scale))
    #         img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
    #         #gridimg = make_grid(img_grid, nrow=1, normalize=False)
    #         #print(gridimg)
    #         #run["imgs/val_img"].upload(gridimg)
    #         #run["valid/misclassified"].log(File(image_path))
    #         save_image(img_grid, "images/validation/%d.png" % batches_done, nrow=1, normalize=False)
    #         run["imgs/val_img"].upload("images/validation/%d.png"% batches_done)
            

            #####################################################################################################
##############################################################################################################
#######################################################################################
######################################################################################################

# # Define model and load model checkpoint
#         vgenerator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
#         vgenerator.load_state_dict(torch.load(opt.checkpoint_model))
#         vgenerator.eval()

#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

#         # Prepare input
#         image_tensor = Variable(transform(Image.open(opt.image_path))).to(device).unsqueeze(0)

#         c = Image.open(opt.image_path)
#         width0, height0 = c.size
#         image_tensor2 = Variable(transform(c.resize((int(width0/2),int(height0/2))))).to(device).unsqueeze(0)
#         # Upsample image
#         with torch.no_grad():
#             sr_image = denormalize(generator(image_tensor2)).cpu()

#         def PSNR(pred, gt, shave_border=0):
#             height, width = pred.shape[2:4]
#             pred = pred[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
#             gt = gt[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
#             psnr = PeakSignalNoiseRatio()
#             psnrprd = psnr (pred,gt)
#             return psnrprd

#         def SSIM(pred, gt, shave_border=0):
#             height, width = pred.shape[2:4]
#             pred = pred[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
#             gt = gt[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
#             print(pred.shape, gt.shape)
#             ssimprd = structural_similarity_index_measure(pred, gt)
#             return ssimprd

#         # #prediction = prediction.cpu()
#         # sr_image2 = sr_image.data[0].numpy().astype(np.float32)
#         # sr_image2 = sr_image2*255.
#         # #sr_image2 = Variable(transform(sr_image2.resize((400,400), Image.ANTIALIAS))).to(device).unsqueeze(0)

#         # image_tensor2 = image_tensor.cpu()
#         # image_tensor2 = image_tensor2.data[0].numpy().astype(np.float32)
#         # image_tensor2 = image_tensor2*255.

#         #print(image_tensor2.shape, sr_image2.shape)

#         psnr_predicted = PSNR(sr_image, image_tensor.type_as(sr_image), shave_border=0)
#         #print("PSNR Predicted = ", psnr_predicted)

#         ssim_predicted = SSIM(sr_image.to(device), image_tensor.to(device), shave_border=0)
#         #print("SSIM Predicted = ", ssim_predicted)
#         all_psnr += psnr_predicted
#         all_ssim += ssim_predicted
        
#     mean_psnr = psnr_predicted/ (i+1)
#     mean_ssim = ssim_predicted/ (i+1)
    
###############################################################################################################
#############################################################################################################33
##########################################################################
#####################################################################################