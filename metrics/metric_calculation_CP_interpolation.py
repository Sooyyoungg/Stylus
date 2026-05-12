
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

def get_unique_parts(data_list):
    """
    Find all unique 'word..._number' combinations from a list of strings
    and return them sorted by numeric value.
    """
    unique_parts = set()

    # Regex pattern: one or more alphanumeric/underscore chars followed by one or more digits.
    # Matches patterns like 'step_water15', 'accordion10', etc.
    pattern = r'[a-zA-Z_]+\d+'

    for item in data_list:
        # Use re.findall to extract all matching substrings.
        found_parts = re.findall(pattern, item)

        # Add to set to automatically remove duplicates.
        unique_parts.update(found_parts)

    # Convert set to list.
    result_list = list(unique_parts)

    # Sort list by numeric value at the end of each string.
    def sort_key(s):
        # Find trailing digits (\d+$) and convert to int.
        match = re.search(r'(\d+)$', s)
        return int(match.group(1)) if match else 0
    
    sorted_list = sorted(result_list, key=sort_key)
    
    return sorted_list


def sort_and_get_unique_prefixes(path_list):
    
    # Sort key function: find the first number in a path and return it as int.
    def get_sort_key(path):
        filename = os.path.basename(path)
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0
    # 1. Sort the original path list.
    sorted_paths = sorted(path_list, key=get_sort_key)

    # 2. Extract prefixes (e.g., 'adventure6', 'adventure22') from each path.
    all_prefixes = [os.path.basename(p).replace(".wav", "").split("_mix")[0] for p in path_list]
    all_prefixes = [string.split("_stylized_with")[0] for string in all_prefixes]
    all_prefixes = get_unique_parts(all_prefixes)
    # 3. Remove duplicates using set, then sort prefixes by their numeric value.
    unique_prefixes_tmp = list(set(all_prefixes))
    unique_prefixes = []
    for prefix in unique_prefixes_tmp: 
        if not "_" in prefix: 
            unique_prefixes.append(prefix)
    sorted_unique_prefixes = sorted(unique_prefixes, key=lambda s: int(re.search(r'\d+', s).group()))
    
    return sorted_paths, sorted_unique_prefixes



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



def get_ContentPreservation_score(ref_dir, file_dir, content=None, style1="accordion", style2='chime',evaluate_mode='reference'):
    #NOTE: Do not need to use style for calculating Content Preservation score
    assert content is not None
    file_list_tmp = glob.glob(f"{file_dir}/*.wav")
    file_list = []
    for i in range(len(file_list_tmp)):
        if content in file_list_tmp[i] and style1 in file_list_tmp[i] and style2 in file_list_tmp[i]:
            file_list.append(file_list_tmp[i])
    file_list_sorted, unique_prefix_list = sort_and_get_unique_prefixes(file_list)

    pairwise_similarities_list = [] 
    for unique_prefix in tqdm(unique_prefix_list): 
        file_list_sub = [] 
        for file in file_list_sorted:
            if f"{unique_prefix}_" in os.path.basename(file):
                file_list_sub.append(file)
        content_file_list = [os.path.join(f"{ref_dir}/{content}", f"{unique_prefix}.wav") for _ in range(len(file_list_sub))] 
        content_audio_embed = [] 
        generated_audio_embed = []
        for content_audio in content_file_list:
            content_audio_embed.append(model.get_audio_embedding_from_filelist(x = [content_audio], use_tensor=True).detach().to("cpu"))
        content_audio_embed = torch.cat(content_audio_embed, dim=0)
        for generated_audio in file_list_sub:
            generated_audio_embed.append(model.get_audio_embedding_from_filelist(x = [generated_audio], use_tensor=True).detach().to("cpu"))
        generated_audio_embed = torch.cat(generated_audio_embed, dim=0)
        pairwise_similarities = torch.nn.functional.cosine_similarity(content_audio_embed, generated_audio_embed, dim=1)
        pairwise_similarities_list.append(pairwise_similarities.mean().item())

    
    return {"content": content, "style1": style1, "style2": style2,"score": np.mean(pairwise_similarities_list).item()}


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
    # Calculate Stylus Content Preservation score
    summary_table_tmp = {"content": [], "style1": [], "style2": [], "score": []}
    for content in content_audio_list:
        for style1, style2 in possible_style_pairs:
            results = get_ContentPreservation_score(ref_dir= ref_dir, file_dir=result_dir, content=content, style1=style1, style2=style2, evaluate_mode='Stylus')
            summary_table_tmp["content"].append(results["content"])
            summary_table_tmp["style1"].append(results["style1"])
            summary_table_tmp["style2"].append(results["style2"])
            summary_table_tmp["score"].append(results["score"])
    summary_table_tmp['evaluate_model'] = ["Stylus" for _ in range(len(summary_table_tmp["score"]))]  # Add evaluate_model column
    summary_table_tmp_df = pd.DataFrame(summary_table_tmp)
    print("Done Stylus Content Preservation Score Calculating...")


    # Save the results
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    summary_table_tmp_df.to_csv(os.path.join(save_dir, f"Content_Preservation_scores_{result_name}_{recon}.csv"), index=False)
    print(f"{result_name}. Content Preservation Score: {summary_table_tmp_df['score'].dropna().values.mean().item()}")





