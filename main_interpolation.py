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

feat_maps = []

import os
import sys
sys.path.append(os.path.abspath("./Stylus/riffusion-hobby"))

#from riffusion.spectrogram_params import SpectrogramParams
from util.riffusion_params import SpectrogramParams
params = SpectrogramParams()

from pathlib import Path
from scipy.io import wavfile
import pydub
import io
import random
def audio_from_waveform(
    samples: np.ndarray, sample_rate: int, normalize: bool = False
) -> pydub.AudioSegment:
    """
    Convert a numpy array of samples of a waveform to an audio segment.

    Args:
        samples: (channels, samples) array
    """
    # Normalize volume to fit in int16
    if normalize:
        samples *= np.iinfo(np.int16).max / np.max(np.abs(samples))

    # Transpose and convert to int16
    samples = samples.transpose(1, 0)
    samples = samples.astype(np.int16)

    # Write to the bytes of a WAV file
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, sample_rate, samples)
    wav_bytes.seek(0)

    # Read into pydub
    return pydub.AudioSegment.from_wav(wav_bytes)
    
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
            # max_iter=params.max_mel_iters,
            # tolerance_loss=1e-5,
            # tolerance_change=1e-8,
            # sgdargs=None,
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
    """
    Compute a spectrogram image from a spectrogram magnitude array.

    This is the inverse of spectrogram_from_image, except for discretization error from
    quantizing to uint8.

    Args:
        spectrogram: (channels, frequency, time)
        power: A power curve to apply to the spectrogram to preserve contrast

    Returns:
        image: (frequency, time, channels)
    """
    # Rescale to 0-1
    max_value = np.max(spectrogram)
    data = spectrogram / max_value

    # Apply the power curve
    data = np.power(data, power)

    # Rescale to 0-255
    data = data * 255

    # Invert
    data = 255 - data

    # Convert to uint8
    data = data.astype(np.uint8)

    # Munge channels into a PIL image
    if data.shape[0] == 1:
        # TODO(hayk): Do we want to write single channel to disk instead?
        image = Image.fromarray(data[0], mode="L").convert("RGB")
    elif data.shape[0] == 2:
        data = np.array([np.zeros_like(data[0]), data[0], data[1]]).transpose(1, 2, 0)
        image = Image.fromarray(data, mode="RGB")
    else:
        raise NotImplementedError(f"Unsupported number of channels: {data.shape[0]}")

    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    return image

def spectrogram_from_image(
    image: Image.Image,
    power: float = 0.25,
    stereo: bool = False,
    max_value: float = 30e6,
) -> np.ndarray:
    """
    Compute a spectrogram magnitude array from a spectrogram image.

    This is the inverse of image_from_spectrogram, except for discretization error from
    quantizing to uint8.

    Args:
        image: (frequency, time, channels)
        power: The power curve applied to the spectrogram
        stereo: Whether the spectrogram encodes stereo data
        max_value: The max value of the original spectrogram. In practice doesn't matter.

    Returns:
        spectrogram: (channels, frequency, time)
    """
    # Convert to RGB if single channel
    if image.mode in ("P", "L"):
        image = image.convert("RGB")

    # Flip Y
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Munge channels into a numpy array of (channels, frequency, time)
    data = np.array(image).transpose(2, 0, 1)
    if stereo:
        # Take the G and B channels as done in image_from_spectrogram
        data = data[[1, 2], :, :]
    else:
        data = data[0:1, :, :]

    # Convert to floats
    data = data.astype(np.float32)

    # Invert
    data = 255 - data

    # Rescale to 0-1
    data = data / 255

    # Reverse the power curve
    data = np.power(data, 1 / power)

    # Rescale to max value
    data = data * max_value

    return data

def save_img_from_sample(model, samples_ddim, fname):
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
    x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
    x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(x_sample.astype(np.uint8))
    img.save(fname)

def feat_merge(opt, cnt_feats, sty1_feats, sty2_feats, start_step=0, num_steps=50):
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'alpha':opt.alpha,
                'mix_beta': opt.mix_beta,
                'T':opt.T,
                'timestep':_,
                }} for _ in range(num_steps)]

    for i in range(len(feat_maps)):
        if i < (num_steps - start_step):
            continue
        cnt_feat = cnt_feats[i]
        sty1_feat = sty1_feats[i]
        sty2_feat = sty2_feats[i]

        for cnt_key, sty1_key, sty2_key in zip(cnt_feat.keys(), sty1_feat.keys(), sty2_feat.keys()):
            if cnt_key[-1] == 'q':
                feat_maps[i][cnt_key] = cnt_feat[cnt_key]
            if cnt_key[-1] == 'k' or cnt_key[-1] == 'v':
                feat_maps[i][f'content_{cnt_key}'] = cnt_feat[cnt_key]
            if sty1_key[-1] == 'k' or sty1_key[-1] == 'v':
                feat_maps[i][f'style1_{sty1_key}'] = sty1_feat[sty1_key]
            if sty2_key[-1] == 'k' or sty2_key[-1] == 'v':
                feat_maps[i][f'style2_{sty2_key}'] = sty2_feat[sty2_key]
    return feat_maps


def load_music(path, output_path):
    segment = AudioSegment.from_file(path)
    sr = segment.frame_rate
    waveform = np.array([c.get_array_of_samples() for c in segment.split_to_mono()])
    
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
    waveform_tensor = torch.from_numpy(waveform)
    
    spectrogram_complex = spectrogram_func(waveform_tensor)
    phase = torch.angle(spectrogram_complex)
    amplitudes = torch.abs(spectrogram_complex)
    # image_s = image_from_spectrogram(amplitudes.numpy(), power=params.power_for_image)
    # image_s.save('./'+path.split('/')[-1].split('.')[0]+'_spect.png')
    
    
    amplitudes_mel = mel_scaler(amplitudes)
    # mel_spectogram = amplitudes_mel.cpu().numpy()
    mel_spectogram = amplitudes_mel.numpy()
    image = image_from_spectrogram(mel_spectogram, power=params.power_for_image)

    # image = Image.open(path).convert("RGB")
    x, y = image.size
    print(f"Loaded input image of size ({x}, {y}) from {path}")
    h = w = 512
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    # image.save('./'+path.split('/')[-1].split('.')[0]+'_mel.png')
    image.save(os.path.join(output_path, 'mel_images', path.split('/')[-1].split('.')[0]+'_mel.png'))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1., phase, sr, x, y

def adain(cnt_feat, sty_feat):
    cnt_mean = cnt_feat.mean(dim=[0, 2, 3],keepdim=True)
    cnt_std = cnt_feat.std(dim=[0, 2, 3],keepdim=True)
    sty_mean = sty_feat.mean(dim=[0, 2, 3],keepdim=True)
    sty_std = sty_feat.std(dim=[0, 2, 3],keepdim=True)
    output = ((cnt_feat-cnt_mean)/cnt_std)*sty_std + sty_mean
    return output

## EFDM
def exact_feature_distribution_matching(content_feat, style_feat):
    assert (content_feat.size() == style_feat.size())
    B, C, W, H = content_feat.size(0), content_feat.size(1), content_feat.size(2), content_feat.size(3)
    value_content, index_content = torch.sort(content_feat.view(B,C,-1))  # sort conduct a deep copy here.
    value_style, _ = torch.sort(style_feat.view(B,C,-1))  # sort conduct a deep copy here.
    inverse_index = index_content.argsort(-1)
    new_content = content_feat.view(B,C,-1) + (value_style.gather(-1, inverse_index) - content_feat.view(B,C,-1).detach())
    return new_content.view(B, C, W, H)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, weights_only=False, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnt', default = './data/cnt')
    parser.add_argument('--sty1', default = './data/sty')
    parser.add_argument('--sty2', default = './data/sty')
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
    parser.add_argument('--alpha', type=float, default=0.75, help='CFG style strength hyperparameter')
    parser.add_argument('--mix_beta', type=float, default=0.75, help='linear interpolation ratio between two style features')
    parser.add_argument("--attn_layer", type=str, default='6,7,8,9,10,11', help='injection attention feature layers')
    parser.add_argument('--model_config', type=str, default='./models/ldm/stable-diffusion-v1/v1-inference.yaml', help='model config')
    parser.add_argument('--precomputed', type=str, default='./precomputed_feats', help='save path for precomputed feature')
    parser.add_argument('--ckpt', type=str, default='./models/ldm/stable-diffusion-v1/model.ckpt', help='model checkpoint')
    parser.add_argument('--precision', type=str, default='autocast', help='choices: ["full", "autocast"]')
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument("--without_init_adain", action='store_true')
    parser.add_argument('--feature_style_matching', type=str, default='adain', help='choices: [adain, efdm]')
    parser.add_argument("--without_attn_injection", action='store_true')
    parser.add_argument('--n_samples_for_ablation', type=int, default=2, help='number of samples for ablation')
    opt = parser.parse_args()

    feat_path_root = opt.precomputed

    seed_everything(22)
    output_path = opt.output_path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path+'/mel_images', exist_ok=True)
    os.makedirs(output_path+'/audios_griffin', exist_ok=True)
    os.makedirs(output_path+'/audios_phase', exist_ok=True)
    if len(feat_path_root) > 0:
        os.makedirs(feat_path_root, exist_ok=True)
    
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    self_attn_output_block_indices = list(map(int, opt.attn_layer.split(',')))
    ddim_inversion_steps = opt.ddim_inv_steps
    save_feature_timesteps = ddim_steps = opt.save_feat_steps

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    unet_model = model.model.diffusion_model
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=opt.ddim_eta, verbose=False) 
    time_range = np.flip(sampler.ddim_timesteps)
    idx_time_dict = {}
    time_idx_dict = {}
    for i, t in enumerate(time_range):
        idx_time_dict[t] = i
        time_idx_dict[i] = t

    seed = torch.initial_seed()
    opt.seed = seed

    global feat_maps
    feat_maps = [{'config': {
                'gamma':opt.gamma,
                'alpha':opt.alpha,
                'T':opt.T
                }} for _ in range(50)]

    def ddim_sampler_callback(pred_x0, xt, i):
        save_feature_maps_callback(i)
        save_feature_map(xt, 'z_enc', i)

    def save_feature_maps(blocks, i, feature_type="input_block"):
        block_idx = 0
        for block_idx, block in enumerate(blocks):
            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                if block_idx in self_attn_output_block_indices:
                    # self-attn
                    q = block[1].transformer_blocks[0].attn1.q
                    k = block[1].transformer_blocks[0].attn1.k
                    v = block[1].transformer_blocks[0].attn1.v
                    save_feature_map(q, f"{feature_type}_{block_idx}_self_attn_q", i)
                    save_feature_map(k, f"{feature_type}_{block_idx}_self_attn_k", i)
                    save_feature_map(v, f"{feature_type}_{block_idx}_self_attn_v", i)
            block_idx += 1

    def save_feature_maps_callback(i):
        save_feature_maps(unet_model.output_blocks , i, "output_block")

    def save_feature_map(feature_map, filename, time):
        global feat_maps
        cur_idx = idx_time_dict[time]
        feat_maps[cur_idx][f"{filename}"] = feature_map

    start_step = opt.start_step
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    uc = model.get_learned_conditioning([""])
    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
    sty1_img_list = sorted(os.listdir(opt.sty1))
    sty2_img_list = sorted(os.listdir(opt.sty2))
    cnt_img_list = sorted(os.listdir(opt.cnt))
    # random sample for ablation 
    if len(sty1_img_list) < opt.n_samples_for_ablation: 
        sty1_img_list = random.sample(sty1_img_list, len(sty1_img_list))
    else: 
        sty1_img_list = random.sample(sty1_img_list, opt.n_samples_for_ablation)
    if len(sty2_img_list) < opt.n_samples_for_ablation: 
        sty2_img_list = random.sample(sty2_img_list, len(sty2_img_list))
    else: 
        sty2_img_list = random.sample(sty2_img_list, opt.n_samples_for_ablation)
    if len(cnt_img_list) < opt.n_samples_for_ablation: 
        cnt_img_list = random.sample(cnt_img_list, len(cnt_img_list))
    else: 
        cnt_img_list = random.sample(cnt_img_list, opt.n_samples_for_ablation)

    total_sample = len(sty2_img_list) * len(sty1_img_list) * len(cnt_img_list) 
    n_sample = 0


    begin = time.time()
    for sty1_name in sty1_img_list:
        sty1_name_ = os.path.join(opt.sty1, sty1_name)
        init_sty1, _, _, _, _ = load_music(sty1_name_, output_path)
        init_sty1 = init_sty1.to(device)
        seed = -1
        sty1_feat_name = os.path.join(feat_path_root, os.path.basename(sty1_name).split('.')[0] + '_sty.pkl')
        sty1_z_enc = None

        # if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
        #     print("Precomputed style feature loading: ", sty_feat_name)
        #     with open(sty_feat_name, 'rb') as h:
        #         sty_feat = pickle.load(h)
        #         sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
        # else:
        init_sty1 = model.get_first_stage_encoding(model.encode_first_stage(init_sty1))
        sty1_z_enc, _ = sampler.encode_ddim(init_sty1.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                            end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                            callback_ddim_timesteps=save_feature_timesteps,
                                            img_callback=ddim_sampler_callback)
        sty1_feat = copy.deepcopy(feat_maps)
        sty1_z_enc = feat_maps[0]['z_enc']

        for sty2_name in sty2_img_list:
            sty2_name_ = os.path.join(opt.sty2, sty2_name)
            init_sty2, _, _, _, _ = load_music(sty2_name_, output_path)
            init_sty2 = init_sty2.to(device)
            seed = -1
            sty2_feat_name = os.path.join(feat_path_root, os.path.basename(sty2_name).split('.')[0] + '_sty.pkl')
            sty2_z_enc = None

            # if len(feat_path_root) > 0 and os.path.isfile(sty_feat_name):
            #     print("Precomputed style feature loading: ", sty_feat_name)
            #     with open(sty_feat_name, 'rb') as h:
            #         sty_feat = pickle.load(h)
            #         sty_z_enc = torch.clone(sty_feat[0]['z_enc'])
            # else:
            init_sty2 = model.get_first_stage_encoding(model.encode_first_stage(init_sty2))
            sty2_z_enc, _ = sampler.encode_ddim(init_sty2.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                callback_ddim_timesteps=save_feature_timesteps,
                                                img_callback=ddim_sampler_callback)
            sty2_feat = copy.deepcopy(feat_maps)
            sty2_z_enc = feat_maps[0]['z_enc']


            for cnt_name in cnt_img_list:
                cnt_name_ = os.path.join(opt.cnt, cnt_name)
                init_cnt, phase, sr, x, y = load_music(cnt_name_, output_path)
                init_cnt = init_cnt.to(device)
                cnt_feat_name = os.path.join(feat_path_root, os.path.basename(cnt_name).split('.')[0] + '_cnt.pkl')
                cnt_feat = None

                # ddim inversion encoding
                # if len(feat_path_root) > 0 and os.path.isfile(cnt_feat_name):
                #     print("Precomputed content feature loading: ", cnt_feat_name)
                #     with open(cnt_feat_name, 'rb') as h:
                #         cnt_feat = pickle.load(h)
                #         cnt_z_enc = torch.clone(cnt_feat[0]['z_enc'])
                # else:
                init_cnt = model.get_first_stage_encoding(model.encode_first_stage(init_cnt))
                cnt_z_enc, _ = sampler.encode_ddim(init_cnt.clone(), num_steps=ddim_inversion_steps, unconditional_conditioning=uc, \
                                                    end_step=time_idx_dict[ddim_inversion_steps-1-start_step], \
                                                    callback_ddim_timesteps=save_feature_timesteps,
                                                    img_callback=ddim_sampler_callback)
                cnt_feat = copy.deepcopy(feat_maps)
                cnt_z_enc = feat_maps[0]['z_enc']

                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            # inversion
                            output_name = f"{os.path.basename(cnt_name).split('.')[0]}_stylized_with_{os.path.basename(sty1_name).split('.')[0]}_and_{os.path.basename(sty2_name).split('.')[0]}"

                            print(f"Inversion end: {time.time() - begin}")
                            if opt.without_init_adain:
                                st_z_enc = cnt_z_enc
                            else:
                                if opt.feature_style_matching== 'adain':
                                    #st_z_enc = adain(cnt_z_enc, sty_z_enc)
                                    raise NotImplementedError("adain is not implemented yet")
                                elif opt.feature_style_matching== 'efdm':
                                    #st_z_enc = exact_feature_distribution_matching(cnt_z_enc, sty_z_enc)
                                    raise NotImplementedError("efdm is not implemented yet")
                            # feat_maps = feat_merge(opt, cnt_feat, sty_feat, start_step=start_step)
                            injected_features = feat_merge(opt, cnt_feat, sty1_feat, sty2_feat, start_step=start_step, num_steps=ddim_inversion_steps)
                            if opt.without_attn_injection:
                                # feat_maps = None
                                injected_features = None
                                

                            # inference
                            samples_ddim, intermediates = sampler.sample(S=ddim_steps,
                                                            batch_size=1,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=st_z_enc,
                                                            # injected_features=feat_maps,
                                                            injected_features=injected_features,
                                                            start_step=start_step,
                                                            )

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                            x_sample = 255. * rearrange(x_image_torch[0].cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            # img.save(os.path.join('./', output_name) + '.png')
                            img.save(os.path.join(output_path, 'mel_images', f'{output_name}_mix{opt.mix_beta}') + '.png')
                            img = img.resize((x, y), resample=Image.Resampling.LANCZOS)
                            
                            mel_spectrogram = spectrogram_from_image(img,
                                        power=params.power_for_image,
                                        stereo=params.stereo,
                                    )
                            amplitudes_mel = torch.from_numpy(mel_spectrogram)
                            # print('amplitudes_mel:', torch.min(amplitudes_mel), torch.max(amplitudes_mel), amplitudes_mel.shape)
                            
                            # MEL SPECTOGRAM -> LINEAR SPECTOGRAM
                            amplitudes_linear = inverse_mel_scaler(amplitudes_mel)

                            # --- Griffin-Lim reconstruction ---
                            waveform = inverse_spectrogram_func(amplitudes_linear.cpu())
                            segment = audio_from_waveform(
                                    samples=waveform.cpu().numpy(),
                                    sample_rate=params.sample_rate,
                                    # Normalize the waveform to the range [-1, 1]
                                    normalize=True,
                                )
                            audio_path = os.path.join(opt.output_path, 'audios_griffin', f'{output_name}_mix{opt.mix_beta}') + '.wav'
                            extension = Path(audio_path).suffix[1:]
                            segment.export(audio_path, format=extension)


                            # --- ISTFT reconstruction ---
                            # Combine generated magnitude with original phase
                            reconstructed_spec = amplitudes_linear * torch.exp(1j * phase)

                            # Perform ISTFT
                            audio = torch.istft(
                                reconstructed_spec.cpu(),
                                n_fft=params.n_fft,
                                hop_length=params.hop_length,
                                win_length=params.win_length,
                                window=torch.hann_window(params.win_length).to("cpu")
                            )

                            # Normalize and save audio
                            audio = audio / torch.max(torch.abs(audio))
                            audio_path = os.path.join(opt.output_path, 'audios_phase', f'{output_name}_mix{opt.mix_beta}') + '.wav'
                            soundfile.write(audio_path, audio.cpu().squeeze().numpy(), sr)

                            
                            # if len(feat_path_root) > 0:
                            #     print("Save features")
                            #     if not os.path.isfile(cnt_feat_name):
                            #         with open(cnt_feat_name, 'wb') as h:
                            #             pickle.dump(cnt_feat, h)
                            #     if not os.path.isfile(sty_feat_name):
                            #         with open(sty_feat_name, 'wb') as h:
                            #             pickle.dump(sty_feat, h)

    print(f"Total end: {time.time() - begin}")

if __name__ == "__main__":
    main()
