### SRGAN model with sliding window, adapted from https://github.com/eriklindernoren/PyTorch-GAN#super-resolution-gan

The training did not finish because the results were not satisfactory. Possible reasons could be:  
 - The way the generator is used to create a single sr-frame from a series of frames. Only the last frame is used in the generator, which was not changed.
 - The batch size was kept small because the weights increased, making the model difficult to fit in the GPU RAM.
