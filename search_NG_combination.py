
import glob 
import os 
import re 
import numpy as np
import librosa
import torch
import laion_clap
from tqdm import tqdm
import pandas as pd


content_audio_list = [
                      "adventure", 
                      "color", 
                      "dance", 
                      "funny_dance", 
                      "hiphop", 
                      "piano", 
                      "relax", 
                      "relieve", 
                      "sad_violin", 
                      "twinkle", 
                      "village", 
                      "violin", 
                      "whistle"
                      ]


style_audio_list = [
                      "accordion", #
                      "chime", #
                      "clarinet", # 
                      "cornet", 
                      "empty", #
                      "fire", #
                      "harp", #
                      "jaw", #
                      "step_water", #
                      "bird", #
                      "church_bell", #
                      "cleanglass", #
                      "cleangalss", # in some case file is not generated well 
                      "drinkglass", #
                      "erhu", #
                      "harmonica", #
                      "heartbeat", #
                      "snow", #
                      "water" #
                      ]



def search_NG_combination(file_dir, content=None, style="accordion", evaluate_mode='reference'):
    assert style is not None, "Style must be specified." 
    assert content is not None
    if evaluate_mode == 'MusicTI' or evaluate_mode == 'MusicGen':
        file_list = glob.glob(file_dir + f"/{style}/{content}/*.wav")
    elif evaluate_mode == 'Stylus':
        file_list_tmp = glob.glob(f"{file_dir}/*.wav")
        text_list_tmp = [os.path.split(file_dir)[-1].replace("_phase.wav", "") for file_dir in file_list_tmp]
        text_list_tmp = [re.sub(r'\d+', '', string).replace("_stylized_", "_") for string in text_list_tmp]
        #text_list_tmp = [string.replace("_stylized_", "_") for string in text_list_tmp]
        file_list = []
        text_list = []
        for i in range(len(text_list_tmp)):
            if content in text_list_tmp[i] and style in text_list_tmp[i]:
                file_list.append(file_list_tmp[i])
                text_list.append(text_list_tmp[i])
                #print(text_list_tmp[i])
        
    if len(file_list) == 0:
        #print(style, content)
        return style, content    
        
    else: 
        return None, None

MusicGen_dir = "./Stylus/MusicTI_AAAI2024/musicgen_transfered_audios"
MusicTI_dir = "./Stylus/MusicTI_AAAI2024/transfered_audios"    
Stylus_dir = "./Stylus/after_phase/stylized_all_gamma0.75_nophase/audios"
save_dir = "./Stylus/StyleID/nonexisting_combination"

# Search Non-existing Combination
summary_table= {"content": [], "style": []}
for content in content_audio_list:
    for style in style_audio_list:
        style_nonexist, content_nonexist = search_NG_combination(file_dir=Stylus_dir, content=content, style=style, evaluate_mode='Stylus')
        summary_table['style'].append(style_nonexist)
        summary_table['content'].append(content_nonexist)
summary_table = pd.DataFrame(summary_table)
summary_table = summary_table.dropna(ignore_index=True).sort_values(by='content')
print(summary_table)
if summary_table.empty:
    print("EVERY COMBINATION IS GENERATED WELL")
else:
    summary_table.to_csv(f"{save_dir}/all_gamma0.5.csv", index=False)
