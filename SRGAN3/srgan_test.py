import argparse

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure

from datasets import TestDataset
from models import GeneratorResNet

import numpy as np
import time
import cv2




parser = argparse.ArgumentParser()

parser.add_argument("--lr_path", type=str,
                    default="", help="directory of lr images to sr")
parser.add_argument("--batch_size", type=int, default=4,
                    help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256,
                    help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256,
                    help="high res. image width")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument("--downsampling", type=int, default=4,
                    help="downsampling of images during training to conver to low resolution")
parser.add_argument("--frames_per_sample", type=int,
                    default=7, help="number of frames per clip sample")
parser.add_argument("--data_root_folder", type=str, default='clips',
                    help="relative path to the txt file that stores the folders paths of the img clips")
parser.add_argument("--resize_hr", type=int, default=1,
                    help="Resize hr images to hr_height and hr_width, 0 is False")
parser.add_argument("--validation_interval", type=int, default=10,
                    help="Number of iterations to update validation losses")
parser.add_argument("--model_path", type=str, default="",
                    help="Path of the saved model as .pth file")



opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'

print(f'Cuda is available: {cuda}')



test_data_loader = DataLoader(
	TestDataset(),
	num_workers = opt.n_cpu,
	batch_size=opt.batch_size,
	shuffle=False
	)


model = GeneratorResNet(opt.channels, opt.channels, n_residual_blocks=16)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def eval():

	# Load model
	model_path = os.path.join(opt.model)
	loadPreTrainedModel(model=model, model_path = model_path)

	model.eval()

	for i, imgs in test_data_loader:


		imgs_lr = Variable(imgs["lr"].type(Tensor))
                        
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        t0 = time.time()

        with torch.no_grad():
	        gen_sr = model(imgs_lr)

       dt = time.time() - t0

       print(f'Processing {i}, {dt}s.')

       save_img(gen_sr, opt.output_path, i)

       gen_sr = gen_sr.cpu()
       gen_sr = gen_sr.data[0].numpy().astype(np.float32)
       gen_sr = gen_sr*255

       imgs_hr = imgs_hr[:,:,-1,:,:].squeeze().numpy().astype(np.float32)
       imgs_hr = imgs_hr*255


       psnr = PSNR(gen_sr, imgs_hr[:,:,-1,:,:])






def save_img(img, output_path, img_count):
	img = img.squeeze().clamp(0,1).numpy().transpose(1,2,0)

	os.mkdir(output_path, exist_ok=True)

	save_fn = output_path + f'/{img_count}.png'

	cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0]) # Whay do bgr to rgb inst of the reverse, from PIL to cv2?


def PSNR(img_sr, img_hr):
    height, width = img_sr.shape[1:] #Batch, Channel, H, W
    img_sr = img_sr[:, 1:height, 1:width]
    img_hr = img_hr[:,1:height, 1:width]
    imdiff = img_sr-img_hr
    rmse = np.sqrt(np.mean(imdiff**2))
    if rmse==0:
        return 100
    return 20*np.log10(255.0/rmse)







