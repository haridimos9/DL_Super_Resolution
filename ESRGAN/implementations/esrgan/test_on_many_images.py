from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from skimage.metrics import structural_similarity
import numpy as np
import math
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics import PeakSignalNoiseRatio

import neptune.new as neptune
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--images_path", type=str, default="splits/val",  help="Path to images")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)



run = neptune.init(
    project="DeepLearningDTU/ESRGAN",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYTIzMTU1ZC1hZDIyLTRkNzAtOTUyYS04MGRhODczYTJlOWYifQ==",
)  # your credentials

run["hyperparameters"] = opt

os.makedirs("images/many_outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=2).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[2:4]
    pred = pred[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    gt = gt[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    psnr = PeakSignalNoiseRatio()
    psnrprd = psnr (pred,gt)
    return psnrprd

def SSIM(pred, gt, shave_border=0):

    height, width = pred.shape[2:4]
    pred = pred[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    gt = gt[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    #print(pred.shape, gt.shape)

    ssimprd = structural_similarity_index_measure(pred, gt)
    return ssimprd

image_list = []
psnr_list = []
ssim_list = []
i=0

for filename in glob.glob(opt.images_path+'/*.png'):
    fname = os.path.basename(filename)
    #im = opt.image_path
    image_list.append(fname)


    # Prepare input
    image_tensor = Variable(transform(Image.open(filename))).to(device).unsqueeze(0)
    c = Image.open(filename)

    width0, height0 = c.size
    image_tensor4 = Variable(transform(c.resize((int(width0/4),int(height0/4))))).to(device).unsqueeze(0)

    #image_tensor4 = Variable(transform(c.resize((125,125), Image.ANTIALIAS))).to(device).unsqueeze(0)
    # Upsample image
    with torch.no_grad():
        sr_image = denormalize(generator(image_tensor4)).cpu()

    # sr_image2 = sr_image.data[0].numpy().astype(np.float32)
    # sr_image2 = sr_image2*255.
    # #sr_image2 = Variable(transform(sr_image2.resize((400,400), Image.ANTIALIAS))).to(device).unsqueeze(0)

    # image_tensor2 = image_tensor.cpu()
    # image_tensor2 = image_tensor2.data[0].numpy().astype(np.float32)
    # image_tensor2 = image_tensor2*255.

    #print(image_tensor2.shape, sr_image2.shape)

    psnr_predicted = PSNR(sr_image, image_tensor.type_as(sr_image), shave_border=0)
    #print("PSNR Predicted = ", psnr_predicted)

    ssim_predicted = SSIM(sr_image, image_tensor.type_as(sr_image), shave_border=0)
    #print("SSIM Predicted = ", ssim_predicted)

    run["testing_step/PSNR"].log(psnr_predicted.item())
    run["testing_step/SSIM"].log(ssim_predicted.item())
    
    psnr_list.append(psnr_predicted.item())
    ssim_list.append(ssim_predicted.item())

    # Save image
    fn = filename.split("/")[-1]
    save_image(sr_image, f"images/many_outputs/sr-{fn}")

    run["imgs/test_img"].upload(f"images/many_outputs/sr-{fn}")

    i+=1

list_of_tuples = list(zip(image_list, psnr_list, ssim_list))
df = pd.DataFrame(list_of_tuples, columns=['image', 'PSNR', 'SSIM'])

df.to_csv('test_metrics.csv')
run["csv/metrics"].upload("test_metrics.csv")
