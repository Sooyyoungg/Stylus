# MusicGen inference time benchmark

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os
import time

model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=5)  # generate 8 seconds.
#wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
descriptions = ['accordion', 'bird', 'chime', 'church_bell',
	'clarinet', 'cleanglass', 'cornet', 'drinkglass', 'empty',
	'erhu', 'fire', 'harmonica', 'harp', 'heartbeat', 'jaw',
	'snow', 'step_water', 'water']

#wav = model.generate(descriptions)  # generates 3 samples.

content_base_dir = './Stylus/audios/content'
content_list = os.listdir(content_base_dir)
for content in content_list:
    file_list = os.listdir(f'{content_base_dir}/{content}')
    for i, file_dir in enumerate(file_list):
        melody, sr = torchaudio.load(f'{content_base_dir}/{content}/{file_dir}')
		# generates using the melody from the given audio and the provided descriptions.
        begin = time.time()
        wav = model.generate_with_chroma(descriptions, melody[None].expand(len(descriptions), -1, -1), sr)
        print(f"Total execution time of MusicGen: {time.time() - begin:.2f} seconds")
        if i == 0:
            break





##TODO code revision for test.


# MusicTI style transfer (run in ldm environment)

import os

base_dir = './Stylus/MusicTI_AAAI2024'
os.chdir(base_dir)
content_list = [folder for folder in os.listdir(f'{base_dir}/audios/audios/content') if '.DS' not in folder]

for style_checkpoint in ['chime_embeddings.pt', 'accordion_embeddings.pt']:
    if style_checkpoint == 'chime_embeddings.pt':
        outdir = f'{base_dir}/outputs_final/chime/'
    elif style_checkpoint == 'accordion_embeddings.pt':
        outdir = f'{base_dir}/outputs_final/accordion/'
    for i, content in enumerate(content_list):
        begin = time.time()
        cmd = (
            f"python scripts/txt2img.py —ddim_eta 0.0 —n_samples 1 —n_iter 2 —scale 5.0 —ddim_steps 50 —strength 0.7 "
            f"—content_path {base_dir}/images/content/{content} "
            f"—embedding_path {base_dir}/logs/{style_checkpoint} "
            f"—ckpt_path {base_dir}/models/ldm/sd/model.ckpt "
            f"—outdir {outdir} "
            f"—prompt '*'"
        )
        os.system(cmd)
        print(f"Total execution time of MusicTI: {time.time() - begin:.2f} seconds")
        if i == 0:
            break
