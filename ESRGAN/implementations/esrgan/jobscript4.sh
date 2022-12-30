 #!/bin/sh
 #BSUB -q gpua100
 #BSUB -gpu "num=1"
 #BSUB -J myJob
 #BSUB -n 1
 #BSUB -W 48:00
 #BSUB -R "rusage[mem=39GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 echo "Running script..."

nvidia-smi

source /work3/s212441/PyTorch-GAN/implementations/esrgan/env/bin/activate

module swap python3/3.9.11
module swap cuda/11.6

python3 /work3/s212441/PyTorch-GAN/implementations/esrgan/esrgan.py --n_epochs 200 --scale 4 --dataset_name "splits/train" --epoch 20 --batch_size 10 --hr_height 512 --hr_width 512 --n_cpu 0 --checkpoint_interval 1000 --warmup_batches 100
