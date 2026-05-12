### Resample & FAD VGG, FAD CLAP

import os 
import glob 
import pandas as pd
import torchaudio
from frechet_audio_distance import FrechetAudioDistance
from tqdm import tqdm

TIMBRE_LIST = [
    'accordion', 
    'chime', 
    'clarinet', 
    'cornet', 
    'empty', 
    'fire', 
    'harp', 
    'jaw', 
    'step_water', 
    'bird', 
    'church_bell', 
    'cleanglass', 
    'cleangalss',  # typo in original code, should be 'cleanglass'
    'drinkglass', 
    'erhu', 
    'harmonica', 
    'heartbeat', 
    'snow' ,
    'water'
    ]
CONTENT_LIST = ['adventure', 'color', 'dance', 'funny_dance', 'hiphop', 'piano', 'relax', 'relieve', 'sad_violin', 'twinkle', 'village', 'violin', 'whistle']


results_dir = 'stylized_CFG_temp1_alpha0.5'
recon = "griffin"  # 'griffin' or 'phase'
EVALUATE_MODEL = 'Stylus'  # 'MusicTI', 'MusicGen', 'Stylus'
CONTENT_REFRENCE_AUDIO_DIRECTORY = './Stylus/audios/content_resampled'
STYLE_REFRENCE_AUDIO_DIRECTORY = './Stylus/audios/timbre_resampled'
STYLIZED_AUDIO_DIRECTORY = f'./Stylus/final_results/{results_dir}/audios_{recon}_resampled'


## calculating FAD

# to use `vggish`
# Vggish model requires all .wav file to have sampling rate of 16kHz
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)



df = {'evaluate_model': [], 'content': [], 'style': [], 'FAD_VGG_content': [], 'FAD_VGG_style': []}


if EVALUATE_MODEL == 'MusicGen':
    for content in tqdm(CONTENT_LIST):
        for timbre in TIMBRE_LIST:
            fad_content_score = frechet.score(
                f"{CONTENT_REFRENCE_AUDIO_DIRECTORY}/{content}", 
                f"{STYLIZED_AUDIO_DIRECTORY}/{timbre}/{content}", 
                dtype="float32"
            )
            fad_style_score = None 
            df['evaluate_model'].append(EVALUATE_MODEL)
            df['content'].append(content)
            df['style'].append(timbre)
            df['FAD_VGG_content'].append(fad_content_score)
            df['FAD_VGG_style'].append(fad_style_score)
else: 
    for content in tqdm(CONTENT_LIST):
        for timbre in TIMBRE_LIST:
            fad_content_score = frechet.score(
                f"{CONTENT_REFRENCE_AUDIO_DIRECTORY}/{content}", 
                f"{STYLIZED_AUDIO_DIRECTORY}/{timbre}/{content}", 
                dtype="float32"
            )
            fad_style_score = frechet.score(
                f"{STYLE_REFRENCE_AUDIO_DIRECTORY}/{timbre}", 
                f"{STYLIZED_AUDIO_DIRECTORY}/{timbre}/{content}", 
                dtype="float32"
            )
            df['evaluate_model'].append(EVALUATE_MODEL)
            df['content'].append(content)
            df['style'].append(timbre)
            df['FAD_VGG_content'].append(fad_content_score)
            df['FAD_VGG_style'].append(fad_style_score)

df = pd.DataFrame(df)
df.to_csv(f'./Stylus/metrics/FAD_VGG_{EVALUATE_MODEL}_{results_dir}_{recon}.csv', index=False)
print(f"FAD_VGG_{EVALUATE_MODEL} score saved to ./Stylus/metrics/FAD_VGG_{EVALUATE_MODEL}_{results_dir}_{recon}.csv")


