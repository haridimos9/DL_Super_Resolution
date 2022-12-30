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

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks, num_upsample=2).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

# Prepare input
image_tensor = Variable(transform(Image.open(opt.image_path))).to(device).unsqueeze(0)

c = Image.open(opt.image_path)
width0, height0 = c.size
image_tensor4 = Variable(transform(c.resize((int(width0/4),int(height0/4))))).to(device).unsqueeze(0)
#image_tensor4 = Variable(transform(c.resize((125,125), Image.ANTIALIAS))).to(device).unsqueeze(0)
# Upsample image
with torch.no_grad():
    sr_image = denormalize(generator(image_tensor4)).cpu()

def PSNR(pred, gt, shave_border=0):
    #height, width = pred.shape[1:3]
    #pred = pred[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    #gt = gt[:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    height, width = pred.shape[2:4]
    pred = pred[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    gt = gt[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    psnr = PeakSignalNoiseRatio()
    psnrprd = psnr (pred,gt)
    #imdff = pred - gt
    #rmse = math.sqrt(np.mean(imdff ** 2))
    #if rmse == 0:
    #    return 100
    #return 20 * math.log10(255.0 / rmse)
    return psnrprd

def SSIM(pred, gt, shave_border=0):
    height, width = pred.shape[2:4]
    pred = pred[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    gt = gt[:,:, 1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    print(pred.shape, gt.shape)
    ssimprd = structural_similarity_index_measure(pred, gt)
    return ssimprd

# #prediction = prediction.cpu()
# sr_image2 = sr_image.data[0].numpy().astype(np.float32)
# sr_image2 = sr_image2*255.
# #sr_image2 = Variable(transform(sr_image2.resize((400,400), Image.ANTIALIAS))).to(device).unsqueeze(0)

# image_tensor2 = image_tensor.cpu()
# image_tensor2 = image_tensor2.data[0].numpy().astype(np.float32)
# image_tensor2 = image_tensor2*255.

#print(image_tensor2.shape, sr_image2.shape)

psnr_predicted = PSNR(sr_image, image_tensor.type_as(sr_image), shave_border=0)
print("PSNR Predicted = ", psnr_predicted)

ssim_predicted = SSIM(sr_image, image_tensor.type_as(sr_image), shave_border=0)
print("SSIM Predicted = ", ssim_predicted)

# Save image
fn = opt.image_path.split("/")[-1]
save_image(sr_image, f"images/outputs/sr-{fn}")
