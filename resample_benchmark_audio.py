import os
import torchaudio
from torchaudio.transforms import Resample

def resample_audio_in_subfolders(src_dir, target_dir, subfolder_list, new_sample_rate=16000):
    """
    Resample .wav files in subfolders of the source directory and save them
    to the target directory with the same folder structure.

    Args:
        src_dir (str): Source directory containing subfolders like 'accordion', 'piano', etc.
                       (e.g., '/.../audios/timbre')
        target_dir (str): Target directory to save the resampled files.
                          (e.g., '/.../audios/timbre_resampled')
        subfolder_list (list): List of subfolder names to process.
        new_sample_rate (int, optional): Target sampling rate. Default is 16000.
    """
    print(f"Starting resampling: [ {os.path.basename(src_dir)} ]")
    print(f"Output location: [ {os.path.basename(target_dir)} ]")
    print(f"Target sampling rate: {new_sample_rate} Hz\n")

    # 1. Create target directory if it does not exist.
    os.makedirs(target_dir, exist_ok=True)

    for folder_name in subfolder_list:
        # 2. Build source and target subfolder paths.
        source_path = os.path.join(src_dir, folder_name)
        target_path = os.path.join(target_dir, folder_name)

        # 3. Check if source directory exists.
        if not os.path.isdir(source_path):
            print(f"Warning: Source directory not found: {source_path}")
            continue

        # 4. Create target subfolder.
        os.makedirs(target_path, exist_ok=True)
        print(f"[{folder_name}] Processing...")

        # 5. Resample files.
        for file_name in os.listdir(source_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(source_path, file_name)

                try:
                    waveform, original_sample_rate = torchaudio.load(file_path)

                    if original_sample_rate != new_sample_rate:
                        resampler = Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
                        resampled_waveform = resampler(waveform)
                    else:
                        resampled_waveform = waveform
                        # Even if sample rate matches, save with '_resampled' suffix for consistency.

                    # Generate new filename and save.
                    base_name, extension = os.path.splitext(file_name)
                    # Skip files already suffixed with '_resampled' to avoid duplicate processing.
                    if base_name.endswith('_resampled'):
                        continue
                    new_file_name = f"{base_name}_resampled{extension}"
                    save_path = os.path.join(target_path, new_file_name)

                    torchaudio.save(save_path, resampled_waveform, new_sample_rate)

                except Exception as e:
                    print(f"  - Error: Failed to process '{file_name}': {e}")

    print("\nAll tasks completed.\n" + "="*40 + "\n")


# --- Configuration and execution ---
if __name__ == "__main__":
    # 1. Absolute paths for source and target directories.
    SOURCE_DIRECTORY = './Stylus/MusicTI_AAAI2024/musicgen_transfered_audios'
    TARGET_DIRECTORY = './Stylus/MusicTI_AAAI2024/musicgen_transfered_audios_resampled'

    # 2. List of subfolders to process.
    TIMBRE_LIST = ['accordion', 'chime', 'clarinet', 'cornet', 'empty', 'fire', 'harp', 'jaw', 'step_water', 'bird', 'church_bell', 'cleanglass', 'drinkglass', 'erhu', 'harmonica', 'heartbeat', 'snow', 'water']
    CONTENT_LIST = ['adventure', 'color','dance', 'funny_dance', 'hiphop', 'piano', 'relax', 'relieve', 'sad_violin', 'twinkle', 'village', 'violin', 'whistle']

    # 3. Target sampling rate.
    TARGET_SAMPLE_RATE = 16000

    # Process files.
    for timbre_name in TIMBRE_LIST:
        if timbre_name == 'cleanglass':
            try:
                timbre_name_tmp = 'cleangalss'  # typo in original code, should be 'cleanglass'
                subfolder_list = [f'{timbre_name_tmp}/{content}' for content in CONTENT_LIST]
                resample_audio_in_subfolders(
                    src_dir=SOURCE_DIRECTORY,
                    target_dir=TARGET_DIRECTORY,
                    subfolder_list=subfolder_list,
                    new_sample_rate=TARGET_SAMPLE_RATE
                )
            except:
                subfolder_list = [f'{timbre_name}/{content}' for content in CONTENT_LIST]
                resample_audio_in_subfolders(
                    src_dir=SOURCE_DIRECTORY,
                    target_dir=TARGET_DIRECTORY,
                    subfolder_list=subfolder_list,
                    new_sample_rate=TARGET_SAMPLE_RATE
                )
        else:
            subfolder_list = [f'{timbre_name}/{content}' for content in CONTENT_LIST]
            resample_audio_in_subfolders(
                src_dir=SOURCE_DIRECTORY,
                target_dir=TARGET_DIRECTORY,
                subfolder_list=subfolder_list,
                new_sample_rate=TARGET_SAMPLE_RATE
            )
