import os
import re
import time
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm
import multiprocessing

def worker_process(args):
    """Worker function that resamples and saves a single file."""
    filename, src_dir, target_dir, timbre_name, content_name, new_sample_rate = args
    target_folder = os.path.join(target_dir, timbre_name, content_name)
    os.makedirs(target_folder, exist_ok=True)
    source_filepath = os.path.join(src_dir, filename)
    target_filepath = os.path.join(target_folder, filename)

    try:
        waveform, original_sr = torchaudio.load(source_filepath)
        if original_sr != new_sample_rate:
            resampler = Resample(orig_freq=original_sr, new_freq=new_sample_rate)
            resampled_waveform = resampler(waveform)
        else:
            resampled_waveform = waveform
        torchaudio.save(target_filepath, resampled_waveform, new_sample_rate)
        return True
    except Exception as e:
        return f"Error: Failed to process '{filename}': {e}"

def resample_and_organize_mp(src_dir, target_dir, timbre_list, content_list, new_sample_rate=16000, num_workers=4, timbre_aliases={}):
    """Resample files in parallel using multiprocessing and organize them."""
    if not os.path.isdir(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        return

    print("Building file map...")
    file_map = {}
    pattern = re.compile(r'^([a-zA-Z_]+)\d+_stylized_([a-zA-Z_]+)\d+\.wav$')
    for filename in os.listdir(src_dir):
        match = pattern.match(filename)
        if match:
            content_name, timbre_name_from_file = match.group(1), match.group(2)
            if timbre_name_from_file not in file_map:
                file_map[timbre_name_from_file] = {}
            if content_name not in file_map[timbre_name_from_file]:
                file_map[timbre_name_from_file][content_name] = []
            file_map[timbre_name_from_file][content_name].append(filename)

    tasks = []
    print("Building task list...")
    for timbre_name in timbre_list:
        # 1. Build list of timbre names to check (official name + aliases).
        names_to_check = [timbre_name] + timbre_aliases.get(timbre_name, [])

        for content_name in content_list:
            files_to_process = []
            # 2. Collect files for the official name and all aliases.
            for name in names_to_check:
                files_to_process.extend(file_map.get(name, {}).get(content_name, []))

            if not files_to_process:
                continue

            for filename in files_to_process:
                # 3. Use the official timbre_name (not alias/typo) when creating tasks.
                tasks.append((filename, src_dir, target_dir, timbre_name, content_name, new_sample_rate))

    if not tasks:
        print("No files found to process.")
        return

    print(f"File map and task list ready. Processing {len(tasks)} files in total.")
    print(f"Starting with {num_workers} worker processes.\n")

    success_count = 0
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Processing files") as pbar:
            for result in pool.imap_unordered(worker_process, tasks):
                if result is True:
                    success_count += 1
                else:
                    print(result)
                pbar.update(1)

    print(f"\nAll tasks completed. Successfully processed {success_count} files.")

# --- Configuration and execution ---
if __name__ == "__main__":
    start_time = time.time()

    # 1. Path configuration.
    results_dir = 'stylized_all_gamma0.75'
    recon = "griffin"  # 'griffin' or 'phase'
    SOURCE_DIRECTORY = f'./Stylus/final_results/{results_dir}/audios_{recon}'
    TARGET_DIRECTORY = f'./Stylus/final_results/{results_dir}/audios_{recon}_resampled'

    # 2. Lists to process.
    TIMBRE_LIST = ['accordion', 'chime', 'clarinet', 'cornet', 'empty', 'fire', 'harp', 'jaw', 'step_water', 'bird', 'church_bell', 'cleanglass', 'drinkglass', 'erhu', 'harmonica', 'heartbeat', 'snow', 'water']
    CONTENT_LIST = ['adventure', 'color', 'dance', 'funny_dance', 'hiphop', 'piano', 'relax', 'relieve', 'sad_violin', 'twinkle', 'village', 'violin', 'whistle']

    # 3. Timbre aliases (e.g., typos in filenames).
    # Format: 'official_name': ['alias1', 'alias2', ...]
    TIMBRE_ALIASES = {
        'cleanglass': ['cleangalss']
        # Add other aliases here if needed, e.g., 'bird': ['birds']
    }

    # 4. Target sampling rate.
    TARGET_SAMPLE_RATE = 16000

    # 5. Number of worker processes.
    NUM_WORKERS = os.cpu_count()

    # Run multiprocessing resampling.
    resample_and_organize_mp(
        src_dir=SOURCE_DIRECTORY,
        target_dir=TARGET_DIRECTORY,
        timbre_list=TIMBRE_LIST,
        content_list=CONTENT_LIST,
        new_sample_rate=TARGET_SAMPLE_RATE,
        num_workers=NUM_WORKERS,
        timbre_aliases=TIMBRE_ALIASES
    )

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
