 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1:mode=exclusive_process"
 #BSUB -J myJob
 #BSUB -n 4
 #BSUB -W 24:00
 #BSUB -R "span[hosts=1]"
 #BSUB -R "rusage[mem=39GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 echo "Running script..."

nvidia-smi
module swap python3/3.9.11
module swap cuda/10.1


source ./venv/bin/activate

python3 srgan.py --data_root_folder 'clips' --n_cpu 4 --sample_interval 20 --checkpoint_interval 50 --hr_height 480 --hr_width 480 --batch_size 2
