
import glob 
import os 
import re 
import numpy as np
import librosa
import torch
import laion_clap
from tqdm import tqdm
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt() # download the default pretrained checkpoint.
model = model.cuda()
#model = model.to("cpu")
# styles="accordion chime clarinet cornet empty fire harp jaw step_water bird church_bell cleanglass drinkglass erhu harmonica heartbeat snow water"
styles="accordion harp jaw empty heartbeat cornet erhu"


content_audio_list = [
                      "color", 
                      "piano", 
                      "relax", 
                      "relieve", 
                      "sad_violin", 
                      "twinkle", 
                      "village", 
                      "violin", 

                      ]


style_audio_list = [
                      "accordion", #
                      "cornet",
                      "empty", #
                      "harp", #
                      "jaw", #
                      "erhu", #
                      "heartbeat", #
                      ]



def get_StyleFit_score(file_dir, content=None, style1="accordion", style2="erhu",evaluate_mode='reference'):
    assert style1 is not None, "Style must be specified." 
    assert style2 is not None, "Style must be specified." 
    
    assert content is not None
    file_list_tmp = glob.glob(f"{file_dir}/*.wav")
    text_list_tmp = [os.path.split(file_dir)[-1].replace("_phase.wav", "").split("_mix")[0] for file_dir in file_list_tmp] 
    text_list_tmp = [re.sub(r'\d+', '', string) for string in text_list_tmp]
    
    file_list = []
    text_list = []
    for i in range(len(text_list_tmp)):
        if content in text_list_tmp[i] and style1 in text_list_tmp[i] and style2 in text_list_tmp[i]:
            file_list.append(file_list_tmp[i])
            style_text_list_tmp = [string.split("_with_")[-1] for string in text_list_tmp]
            sty1, sty2 = style_text_list_tmp[i].split("_and_")
            text_list.append((sty1, sty2))

    audio_embed = []
    bsz = 1 
    for i in range(0, len(file_list), bsz):
        data = file_list[i:i+bsz]
        embed = model.get_audio_embedding_from_filelist(x = data, use_tensor=True)
        audio_embed.append(embed.detach().to("cpu"))
    #print(f"content: {content}, style: {style}, len: {len(audio_embed)}")
    if len(audio_embed) > 0:
        audio_embed = torch.cat(audio_embed, dim=0)
        #audio_embed = model.get_audio_embedding_from_filelist(x = file_list, use_tensor=True)
        sty1_text_embed = []
        sty2_text_embed = []
        for sty1, sty2 in text_list:
            sty1_text_embed.append(model.get_text_embedding(sty1))
            sty2_text_embed.append(model.get_text_embedding(sty2))
        sty1_text_embed = torch.tensor(sty1_text_embed).squeeze(1)
        sty2_text_embed = torch.tensor(sty2_text_embed).squeeze(1)
        sty1_pairwise_similarities = torch.nn.functional.cosine_similarity(audio_embed.to(sty1_text_embed.device), sty1_text_embed, dim=1).mean().item()
        sty2_pairwise_similarities = torch.nn.functional.cosine_similarity(audio_embed.to(sty2_text_embed.device), sty2_text_embed, dim=1).mean().item()
    else:
        sty1_pairwise_similarities = None
        sty2_pairwise_similarities = None

    return {"content": content, "style1": style1, "style2": style2, "style1_score": sty1_pairwise_similarities,  "style2_score": sty2_pairwise_similarities}



ablation_idx = 'ablation5'
recon = "phase"  # 'griffin' or 'phase'
ref_dir = "./Stylus/audios/content"
result_list = [["default_beta0.1"], "default_beta0.3", "default_beta0.5", "default_beta0.7", "default_beta0.9"] 
#result_list = ["default_beta0.1"]
result_dir = [f"./Stylus/final_results/{ablation_idx}/{result}/audios_{recon}" for result in result_list]
save_dir = f"./Stylus/metrics/{ablation_idx}"



musical = ["accordion","chime","clarinet","cornet","harp","church_bell","erhu","harmonica",]
non_musical = ["empty","fire","jaw","step_water","bird","cleangalss", "cleanglass", "drinkglass","heartbeat","snow","water" ]


possible_style_pairs = list(combinations(style_audio_list, 2))



for result_name, result_dir in zip(result_list, result_dir):
    # Calculate Stylus Style Fit score
    summary_table_tmp = {"content": [], "style1": [], "style2": [], "style1_score": [], "style2_score": []}
    for content in content_audio_list:
        for style1, style2 in possible_style_pairs:
            results = get_StyleFit_score(file_dir=result_dir, content=content, style1=style1, style2=style2, evaluate_mode='Stylus')
            summary_table_tmp["content"].append(results["content"])
            summary_table_tmp["style1"].append(results["style1"])
            summary_table_tmp["style2"].append(results["style2"])
            summary_table_tmp["style1_score"].append(results["style1_score"])
            summary_table_tmp["style2_score"].append(results["style2_score"])
    summary_table_tmp['evaluate_model'] = ["Stylus" for _ in range(len(summary_table_tmp["style1_score"]))]  # Add evaluate_model column
    summary_table_tmp_df = pd.DataFrame(summary_table_tmp)
    print("Done Stylus Style Fit Score Calculating...")


    # Save the results
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    summary_table_tmp_df.to_csv(os.path.join(save_dir, f"StyleFit_scores_{result_name}_{recon}.csv"), index=False)

    print(f"{result_name}. Style1-Style Fit Score: {summary_table_tmp_df['style1_score'].dropna().values.mean().item()}, Style2-Style Fit Score: {summary_table_tmp_df['style2_score'].dropna().values.mean().item()}")





