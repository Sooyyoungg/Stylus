import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
import copy

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import torchvision.transforms as transforms
import torch.nn.functional as F
import time
import pickle

from pydub import AudioSegment
import torchaudio
import cv2
import soundfile

# --- Additional imports for Multi-GPU/Multi-Node ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

feat_maps = []

import os
import sys
sys.path.append(os.path.abspath("./Stylus/riffusion-hobby"))

from util.riffusion_params import SpectrogramParams
params = SpectrogramParams()

from pathlib import Path
from scipy.io import wavfile
import pydub
import io
def audio_from_waveform(
    samples: np.ndarray, sample_rate: int, normalize: bool = False
) -> pydub.AudioSegment:
    if normalize:
        samples *= np.iinfo(np.int16).max / np.max(np.abs(samples))
    samples = samples.transpose(1, 0)
    samples = samples.astype(np.int16)
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples)
    wav_bytes.seek(0)
    return pydub.AudioSegment.from_wav(wav_bytes)

# --- Global transform objects ---
spectrogram_func = torchaudio.transforms.Spectrogram(
        n_fft=params.n_fft,
        hop_length=params.hop_length,
        win_length=params.win_length,
        pad=0,
        window_fn=torch.hann_window,
        power=None,
        normalized=False,
        wkwargs=None,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )
mel_scaler = torchaudio.transforms.MelScale(
        n_mels=params.num_frequencies,
        sample_rate=params.sample_rate,
        f_min=params.min_frequency,
        f_max=params.max_frequency,
        n_stft=params.n_fft // 2 + 1,
        norm=params.mel_scale_norm,
        mel_scale=params.mel_scale_type,
    )

inverse_mel_scaler = torchaudio.transforms.InverseMelScale(
            n_stft=params.n_fft // 2 + 1,
            n_mels=params.num_frequencies,
            sample_rate=params.sample_rate,
            f_min=params.min_frequency,
            f_max=params.max_frequency,
            norm=params.mel_scale_norm,
            mel_scale=params.mel_scale_type,
        )

inverse_spectrogram_func = torchaudio.transforms.GriffinLim(
            n_fft=params.n_fft,
            n_iter=params.num_griffin_lim_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            window_fn=torch.hann_window,
            power=1.0,
            wkwargs=None,
            momentum=0.99,
            length=None,
            rand_init=True,
        )

def image_from_spectrogram(spectrogram: np.ndarray, power: float = 0.25) -> Image.Image:
    max_value = np.max(spectrogram)
    data = spectrogram / max_value
    data = np.power(data, power)
    data = data * 255
    data = 255 - data
    data = data.astype(np.uint8)
    if data.shape[0] == 1:
        image = Image.fromarray(data[0], mode="L").convert("RGB")
    elif data.shape[0] == 2:
        data = np.array([np.zeros_like(data[0]), data[0], data[1]]).transpose(1, 2, 0)
        image = Image.fromarray(data, mode="RGB")
    else:
        raise NotImplementedError(f"Unsupported number of channels: {data.shape[0]}")
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image

def spectrogram_from_image(
    image: Image.Image,
    power: float = 0.25,
    stereo: bool = False,
    max_value: float = 30e6,
) -> np.ndarray:
    if image.mode in ("P", "L"):
        image = image.convert("RGB")
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    data = np.array(image).transpose(2, 0, 1)
    if stereo:
        data = data[[1, 2], :, :]
    else:
        data = data[0:1, :, :]
    data = data.astype(np.float32)
    data = 255 - data
    data = data / 255
    data = np.power(data, 1 / power)
    data = data * max_value
    return data

def save_img_from_sample(model, samples_ddim, fname):
    raw_model = model.module if isinstance(model, DDP) else model
    x_samples_ddim = raw_model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)

def feat_merge(opt, cnt_feats, sty_feats, start_step=0, num_steps=50):
    feat_maps = [{'config': {'gamma':opt.gamma, 'T':opt.T, 'timestep':_}} for _ in range(num_steps)]
    for i in range(len(feat_maps)):
        if i < (num_steps - start_step): continue
        cnt_feat, sty_feat = cnt_feats[i], sty_feats[i]
        for ori_key in sty_feat.keys():
            if ori_key[-1] == 'q': feat_maps[i][ori_key] = cnt_feat[ori_key]
            if ori_key[-1] == 'k' or ori_key[-1] == 'v': feat_maps[i][ori_key] = sty_feat[ori_key]
    return feat_maps

def load_music(path, device):
    segment = AudioSegment.from_file(path)
    sr = segment.frame_rate
    waveform = np.array([c.get_array_of_samples() for c in segment.split_to_mono()])
    if waveform.dtype != np.float32: waveform = waveform.astype(np.float32)
    waveform_tensor = torch.from_numpy(waveform).to(device)
    
    spectrogram_complex = spectrogram_func(waveform_tensor)
    phase = torch.angle(spectrogram_complex)
    amplitudes = torch.abs(spectrogram_complex)

    amplitudes_mel = mel_scaler(amplitudes)
    mel_spectogram = amplitudes_mel.cpu().numpy()
    image = image_from_spectrogram(mel_spectogram, power=params.power_for_image)
    
    x, y = image.size
    h = w = 512
    image_cropped = transforms.CenterCrop(min(x,y))(image)
    image_resized = image_cropped.resize((w, h), resample=Image.Resampling.LANCZOS)
    
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_np = image_np[None].transpose(0, 3, 1, 2)
    image_tensor = torch.from_numpy(image_np)
    return 2.*image_tensor - 1., phase, sr, x, y, image_resized

def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output

def exact_feature_distribution_matching(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
    value_content, index_content = torch.sort(content_feat.view(B,C,-1))
    value_style, _ = torch.sort(style_feat.view(B,C,-1))
    inverse_index = index_content.argsort(-1)
    new_content = content_feat.view(B,C,-1) + (value_style.gather(-1, inverse_index) - content_feat.view(B,C,-1).detach())
    return new_content.view(B, C, W, H)

def load_model_from_config(config, ckpt, verbose=False, device='cuda'):
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    if is_main_process: print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, weights_only=False, map_location="cpu")
    if is_main_process and "global_step" in pl_sd: print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if is_main_process and verbose:
        if len(m) > 0: print("missing keys:", m)
        if len(u) > 0: print("unexpected keys:", u)
    model.to(device)
    model.eval()
    return model

def setup_distributed():
    if not dist.is_available() or "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) == 1:
        return 0, 1
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    print(f"Initialized DDP: RANK {rank}, WORLD_SIZE {world_size}, LOCAL_RANK {local_rank}")
    return rank, world_size

def cleanup():
    if dist.is_initialized(): dist.destroy_process_group()

def main():
    rank, world_size = setup_distributed()
    is_main_process = (rank == 0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = './data/cnt')
    parser.add_argument('--sty', default = './data/sty')
    parser.add_argument('--ddim_inv_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--save_feat_steps', type=int, default=50, help='DDIM eta')
    parser.add_argument('--start_step', type=int, default=49, help='DDIM eta')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='DDIM eta')
    parser.add_argument('--H', type=int, default=512, help='image height, in pixel space')
    parser.add_argument('--W', type=int, default=512, help='image width, in pixel space')
    parser.add_argument('--C', type=int, default=4, help='latent channels')
    parser.add_argument('--f', type=int, default=8, help='downsampling factor')
    parser.add_argument('--T', type=float, default=1.5, help='attention temperature scaling hyperparameter')
    parser.add_argument('--gamma', type=float, default=0.75, help='query preservation hyperparameter')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='./models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats', help='save path for precomputed feature')
    parser.add_argument('--ckpt', type=str, default='./models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument('--feature_style_matching', type=str, default='adain', help='choices: [adain, efdm]')
    parser.add_argument("--without_attn_injection", action='store_true')
    opt = parser.parse_args()

    if is_main_process:
        os.makedirs(opt.output_path, exist_ok=True)
        os.makedirs(os.path.join(opt.output_path, 'mel_images'), exist_ok=True)
        os.makedirs(os.path.join(opt.output_path, 'audios'), exist_ok=True)
    
    seed_everything(42 + rank)
    
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    
    spectrogram_func.to(device)
    mel_scaler.to(device)
    inverse_mel_scaler.to(device)
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}", device=device)

    if world_size > 1:
        model = DDP(model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)

    raw_model = model.module if world_size > 1 else model
    
    unet_model = raw_model.model.diffusion_model
    sampler = DDIMSampler(raw_model)
    assert opt.ddim_inv_steps == opt.save_feat_steps, "DDIM inversion steps and feature saving steps must be the same."
    start_step = opt.ddim_inv_steps - 1
    sampler.make_schedule(ddim_num_steps=opt.ddim_inv_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {t: i for i, t in enumerate(time_range)}
    time_idx_dict = {i: t for i, t in enumerate(time_range)}

    global feat_maps
    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)
    def save_feature_maps(blocks, i, feature_type="input_block"):
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    q, k, v = block[1].transformer_blocks[0].attn1.q, block[1].transformer_blocks[0].attn1.k, block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks, i, "output_block")
    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map
    
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    uc = raw_model.get_learned_conditioning([""]).to(device)
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

    sty_img_list = sorted(os.listdir(opt.sty))
    cnt_img_list = sorted(os.listdir(opt.cnt))
    all_pairs = [(sty, cnt) for sty in sty_img_list for cnt in cnt_img_list]
    
    pairs_for_this_rank = all_pairs[rank::world_size]
    
    if is_main_process:
        print(f"Total pairs: {len(all_pairs)}. Each of {world_size} processes will handle ~{len(pairs_for_this_rank)} pairs.")

    begin = time.time()
    for n_sample, (sty_name, cnt_name) in enumerate(pairs_for_this_rank):
        if is_main_process:
            print(f"Processing pair {n_sample+1}/{len(pairs_for_this_rank)} on Rank {rank}: Style='{sty_name}', Content='{cnt_name}'")

        feat_maps = [{'config': {'gamma': opt.gamma, 'T': opt.T}} for _ in range(50)]

        sty_name_ = os.path.join(opt.sty, sty_name)
        init_sty, _, _, _, _, _ = load_music(sty_name_, device)
        init_sty = init_sty.to(device)
        
        init_sty_enc = raw_model.get_first_stage_encoding(raw_model.encode_first_stage(init_sty))
        _, _ = sampler.encode_ddim(init_sty_enc.clone(), num_steps=opt.ddim_inv_steps, unconditional_conditioning=uc,
                                   end_step=time_idx_dict[opt.ddim_inv_steps - 1 - start_step],
                                   callback_ddim_timesteps=opt.save_feat_steps, img_callback=ddim_sampler_callback)
        sty_feat = copy.deepcopy(feat_maps)
        sty_z_enc = feat_maps[0]['z_enc']

        cnt_name_ = os.path.join(opt.cnt, cnt_name)
        init_cnt, phase, sr, x, y, resized_mel_img = load_music(cnt_name_, device)
        init_cnt = init_cnt.to(device)
        phase = phase.to(device)
        
        init_cnt_enc = raw_model.get_first_stage_encoding(raw_model.encode_first_stage(init_cnt))
        _, _ = sampler.encode_ddim(init_cnt_enc.clone(), num_steps=opt.ddim_inv_steps, unconditional_conditioning=uc,
                                   end_step=time_idx_dict[opt.ddim_inv_steps - 1 - start_step],
                                   callback_ddim_timesteps=opt.save_feat_steps, img_callback=ddim_sampler_callback)
        cnt_feat = copy.deepcopy(feat_maps)
        cnt_z_enc = feat_maps[0]['z_enc']
        
        if is_main_process:
             resized_mel_img.save(os.path.join(opt.output_path, 'mel_images', cnt_name.split('.')[0]+'_mel.png'))

        with torch.no_grad():
            with precision_scope("cuda"):
                with raw_model.ema_scope():
                    output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_{os.path.basename(sty_name).split('.')[0]}"
                    
                    if opt.without_init_adain:
                        st_z_enc = cnt_z_enc
                    else:
                        st_z_enc = adain(cnt_z_enc, sty_z_enc) if opt.feature_style_matching == 'adain' else exact_feature_distribution_matching(cnt_z_enc, sty_z_enc)

                    injected_features = None if opt.without_attn_injection else feat_merge(opt, cnt_feat, sty_feat, start_step=start_step, num_steps=opt.save_feat_steps)
                    
                    samples_ddim, _ = sampler.sample(S=opt.save_feat_steps, batch_size=1, shape=shape, verbose=False,
                                                     unconditional_conditioning=uc, eta=opt.ddim_eta, x_T=st_z_enc,
                                                     injected_features=injected_features, start_step=start_step)

                    x_samples_ddim = raw_model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    x_sample_np = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_image_torch = torch.from_numpy(x_sample_np).permute(0, 3, 1, 2)
                    x_sample_img_np = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample_img_np.astype(np.uint8))
                    
                    img.save(os.path.join(opt.output_path, 'mel_images', output_name) + '.png')
                    img = img.resize((x, y), resample=Image.Resampling.LANCZOS)
                    
                    mel_spectrogram = spectrogram_from_image(img, power=params.power_for_image, stereo=params.stereo)
                    amplitudes_mel = torch.from_numpy(mel_spectrogram).to(device)
                    amplitudes_linear = inverse_mel_scaler(amplitudes_mel)

                    
                    # --- Griffin-Lim reconstruction ---

                    waveform = inverse_spectrogram_func(amplitudes_linear.cpu())
                    segment = audio_from_waveform(
                            samples=waveform.cpu().numpy(),
                            sample_rate=params.sample_rate,
                            # Normalize the waveform to the range [-1, 1]
                            normalize=True,
                        )
                    audio_path = os.path.join(opt.output_path, 'audios', output_name) + '_nophase.wav'
                    extension = Path(audio_path).suffix[1:]
                    segment.export(audio_path, format=extension)
                    """
                    # --- ISTFT reconstruction ---
                    # Combine generated magnitude with original phase
                    reconstructed_spec = amplitudes_linear * torch.exp(1j * phase)

                    # Perform ISTFT
                    audio = torch.istft(
                        reconstructed_spec,
                        n_fft=params.n_fft,
                        hop_length=params.hop_length,
                        win_length=params.win_length,
                        window=torch.hann_window(params.win_length).to(device)
                    )

                    # Normalize and save audio
                    audio = audio / torch.max(torch.abs(audio))
                    audio_path = os.path.join(opt.output_path, 'audios', output_name) + '_phase.wav'
                    soundfile.write(audio_path, audio.cpu().squeeze().numpy(), sr)
                    """

    if world_size > 1: dist.barrier()

    if is_main_process:
        print(f"Total execution time: {time.time() - begin:.2f} seconds")
    
    cleanup()

if __name__ == "__main__":
    main()
