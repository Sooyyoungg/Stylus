
import glob 
import os 
import re 
import numpy as np
import librosa
import torch
import laion_clap
from tqdm import tqdm
import pandas as pd

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt() # download the default pretrained checkpoint.
model = model.cuda()
#model = model.to("cpu")


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
                      "cleangalss",
                      "drinkglass", #
                      "erhu", #
                      "harmonica", #
                      "heartbeat", #
                      "snow", #
                      "water" #
                      ]



def get_StyleFit_score(file_dir, content=None, style="accordion", evaluate_mode='reference'):
    assert style is not None, "Style must be specified." 
    if content is None:
        assert evaluate_mode == 'reference'
        file_list = glob.glob(file_dir + f"/{style}/*.wav")
        text_list = [os.path.split(file_dir)[-1].replace(".wav", "") for file_dir in file_list] 
        text_list = [re.sub(r'\d+', '', string).replace("_", " ") for string in text_list]
    else:
        assert content is not None
        if evaluate_mode == 'MusicTI' or evaluate_mode == 'MusicGen':
            file_list = glob.glob(file_dir + f"/{style}/{content}/*.wav")
            text_list = [f"{style}".replace("_", " ") for _ in range(len(file_list))]
        elif evaluate_mode == 'Stylus':
            file_list_tmp = glob.glob(f"{file_dir}/*.wav")
            text_list_tmp = [os.path.split(file_dir)[-1].replace("_phase.wav", "") for file_dir in file_list_tmp] 
            text_list_tmp = [re.sub(r'\d+', '', string).replace("_stylized_", "_") for string in text_list_tmp]
            file_list = []
            text_list = []
            for i in range(len(text_list_tmp)):
                if content in text_list_tmp[i] and style in text_list_tmp[i]:
                    file_list.append(file_list_tmp[i])
                    text_list.append(text_list_tmp[i])
            text_list = [string.split("_")[-1] for string in text_list]

    audio_embed = []
    bsz = 1 
    for i in range(0, len(file_list), bsz):
        data = file_list[i:i+bsz]
        embed = model.get_audio_embedding_from_filelist(x = data, use_tensor=True)
        audio_embed.append(embed.detach().to("cpu"))
    print(f"content: {content}, style: {style}, len: {len(audio_embed)}")
    if len(audio_embed) > 0:
        audio_embed = torch.cat(audio_embed, dim=0)
        #audio_embed = model.get_audio_embedding_from_filelist(x = file_list, use_tensor=True)
        text_embed = [model.get_text_embedding(text) for text in text_list]
        text_embed = torch.tensor(text_embed).squeeze(1)
        pairwise_similarities = torch.nn.functional.cosine_similarity(audio_embed.to(text_embed.device), text_embed, dim=1).mean().item()
    else:
        pairwise_similarities = None
    return {"content": content, "style": style, "score": pairwise_similarities}


results_dir = "stylized_all_gamma0.75_nophase"
recon = "griffin"  # 'griffin' or 'phase'
ref_dir = "./Stylus/audios/timbre"
MusicGen_dir = "./Stylus/MusicTI_AAAI2024/musicgen_transfered_audios"
MusicTI_dir = "./Stylus/MusicTI_AAAI2024/transfered_audios"    
Stylus_dir = f"./Stylus/final_results/{results_dir}/audios_{recon}"
save_dir = "./Stylus/metrics"

summary_table = {
                 'evaluate_model': [],
                 'content': [],
                 'style': [],
                 'score': []
                 }
summary_table_df = pd.DataFrame(summary_table)

"""
# Calculate Reference Style Fit score
summary_table_tmp = {"content": [], "style": [], "score": []}
for style in style_audio_list:
    results = get_StyleFit_score(file_dir=reference_dir, style=style, evaluate_mode='reference')
    summary_table_tmp["content"].append(None)
    summary_table_tmp["style"].append(results["style"])
    summary_table_tmp["score"].append(results["score"])
summary_table_tmp['evaluate_model'] = ["reference" for _ in range(len(summary_table_tmp["score"]))]  # Add evaluate_model column
summary_table_tmp_df = pd.DataFrame(summary_table_tmp)
summary_table_df = pd.concat([summary_table_df, summary_table_tmp_df], ignore_index=True)
print("Done Reference Style Fit Score Calculating...")

# Calculate MusicGen Style Fit score
summary_table_tmp = {"content": [], "style": [], "score": []}
for content in content_audio_list:
    for style in style_audio_list:
        results = get_StyleFit_score(file_dir=MusicGen_dir, content=content, style=style, evaluate_mode='MusicGen')
        summary_table_tmp["content"].append(results["content"])
        summary_table_tmp["style"].append(results["style"])
        summary_table_tmp["score"].append(results["score"])
summary_table_tmp['evaluate_model'] = ["MusicGen" for _ in range(len(summary_table_tmp["score"]))]  # Add evaluate_model column
summary_table_tmp_df = pd.DataFrame(summary_table_tmp)
summary_table_df = pd.concat([summary_table_df, summary_table_tmp_df], ignore_index=True)
print("Done MusicGen Style Fit Score Calculating...")


# Calculate MusicTI Style Fit score
summary_table_tmp = {"content": [], "style": [], "score": []}
for content in content_audio_list:
    for style in style_audio_list:
        results = get_StyleFit_score(file_dir=MusicTI_dir, content=content, style=style, evaluate_mode='MusicTI')
        summary_table_tmp["content"].append(results["content"])
        summary_table_tmp["style"].append(results["style"])
        summary_table_tmp["score"].append(results["score"])
summary_table_tmp['evaluate_model'] = ["MusicTI" for _ in range(len(summary_table_tmp["score"]))]  # Add evaluate_model column
summary_table_tmp_df = pd.DataFrame(summary_table_tmp)
summary_table_df = pd.concat([summary_table_df, summary_table_tmp_df], ignore_index=True)
print("Done MusicTI Style Fit Score Calculating...")
"""

# Calculate Stylus Style Fit score
summary_table_tmp = {"content": [], "style": [], "score": []}
for content in content_audio_list:
    for style in style_audio_list:
        results = get_StyleFit_score(file_dir=Stylus_dir, content=content, style=style, evaluate_mode='Stylus')
        summary_table_tmp["content"].append(results["content"])
        summary_table_tmp["style"].append(results["style"])
        summary_table_tmp["score"].append(results["score"])
summary_table_tmp['evaluate_model'] = ["Stylus" for _ in range(len(summary_table_tmp["score"]))]  # Add evaluate_model column
summary_table_tmp_df = pd.DataFrame(summary_table_tmp)
summary_table_df = pd.concat([summary_table_df, summary_table_tmp_df], ignore_index=True)
print("Done Stylus Style Fit Score Calculating...")


# Save the results
summary_table_df.to_csv(os.path.join(save_dir, f"StyleFit_scores_{results_dir}_{recon}.csv"), index=False)
