#!/bin/bash

module load python 
conda activate stylus

export HF_HOME={huggingface_dir}
export TORCH_HOME={torch_dir}

data_dir="./Stylus/audios"
out_dir="./Stylus/results/main"

contents="adventure color dance funny_dance hiphop piano relax relieve sad_violin twinkle village violin whistle"
styles="accordion chime clarinet cornet empty fire harp jaw step_water bird church_bell cleanglass drinkglass erhu harmonica heartbeat snow water"

pairs=()

for i in ${styles[@]}; do
  for j in ${styles[@]}; do
    if [[ "$i" < "$j" ]]; then
      pairs+=("$i,$j")
    fi
  done
done


for cont in $contents
do 
    for p in ${pairs[@]}
    do
        style1=$(echo "$p" | cut -d',' -f1)
        style2=$(echo "$p" | cut -d',' -f2)
        torchrun --nproc_per_node=1 ./Stylus/main_interpolation.py --gamma 0.75 --alpha 1 --mix_beta 0.5 --n_samples_for_ablation 2 --without_init_adain --ddim_inv_steps 50 --save_feat_steps 50 --cnt "$data_dir/content/$cont" --sty1 "$data_dir/timbre/$style1" --sty2 "$data_dir/timbre/$style2" --output_path $out_dir
    done
done

