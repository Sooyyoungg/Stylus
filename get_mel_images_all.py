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
import glob

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
    image.save(output_path)
    #image.save(os.path.join(output_path,path.split('/')[-1].split('.')[0]+'_mel.png'))




input_dir = "./Stylus/asset/MusicGen_audios"
output_dir = "./Stylus/asset/MusicGen_imgs"

input_file_list = glob.glob(os.path.join(input_dir, '*.wav'))
file_name = [os.path.split(input_file)[-1].replace(".wav", "") for input_file in input_file_list]
output_file_list = [f"{output_dir}/{output_file}.png" for output_file in file_name]

for input_path, output_path in zip(input_file_list, output_file_list): 
    load_music(input_path, output_path)
