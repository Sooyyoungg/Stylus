#!/bin/bash
#SBATCH -A {gpu}
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 15:00:00
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --chdir=./Stylus
#SBATCH --exclusive
#SBATCH --output=./Stylus/logs/R_CFG_temp1_alpha0.9-%j-%x.out

module load python 
conda activate stylus

export HF_HOME={huggingface_dir}
export TORCH_HOME={torch_dir}

data_dir="./Stylus/audios"
out_dir="./Stylus/results/main"

contents="adventure color dance funny_dance hiphop piano relax relieve sad_violin twinkle village violin whistle"
styles="accordion chime clarinet cornet empty fire harp jaw step_water bird church_bell cleanglass drinkglass erhu harmonica heartbeat snow water"

for cont in $contents
do
    for sty in $styles
    do
        torchrun --nproc_per_node=4 ./Stylus/main.py --gamma 0.75 --temperature 1 --alpha 0.9 --ddim_inv_steps 50 --save_feat_steps 50 --cnt "$data_dir/content/$cont" --sty "$data_dir/timbre/$sty" --output_path $out_dir
    done
done
 
