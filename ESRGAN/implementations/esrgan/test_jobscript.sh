 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J myJob
 #BSUB -n 1
 #BSUB -W 24:00
 #BSUB -R "rusage[mem=10GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 echo "Running script..."

nvidia-smi

module swap python3/3.9.11
module swap cuda/11.6

source ./env/bin/activate

python3 test_on_imageupscale.py --image_path "/work3/s212441/ESRGAN/94.png" --checkpoint_model "/work3/s212441/ESRGAN/implementations/esrgan/saved_models/generator_32.pth"
