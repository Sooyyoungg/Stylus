import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os

def apply_magma_colormap(input_path: str, output_path: str):
    """
    Load a grayscale image, apply the 'magma' colormap, and save the result.

    Args:
        input_path (str): Path to the input grayscale image file.
        output_path (str): Path to save the output color image file.
    """
    try:
        # 1. Load grayscale image using PIL.
        # .convert('L') ensures the image is a single-channel (Luminance) image.
        gray_image_pil = Image.open(input_path).convert('L')

        # 2. Convert image to NumPy array (integer values in range 0-255).
        gray_array = np.array(gray_image_pil)

        # 3. Get the 'magma' colormap from Matplotlib.
        cmap = plt.get_cmap('magma')

        # 4. Normalize data to range 0.0 ~ 1.0 for colormap application.
        normalized_array = gray_array / 255.0

        # 5. Apply colormap (result has RGBA values in range 0.0 ~ 1.0).
        colored_array_rgba = cmap(normalized_array)

        # 6. Convert to 8-bit RGB format (range 0 ~ 255) for saving with PIL.
        #    Alpha (A) channel is excluded.
        colored_array_rgb_uint8 = (colored_array_rgba[:, :, :3] * 255).astype(np.uint8)

        # 7. Convert NumPy array back to PIL image.
        colored_image_pil = Image.fromarray(colored_array_rgb_uint8)

        # 8. Save the final color image.
        colored_image_pil.save(output_path)

        print(f"Success: '{input_path}' converted and saved to '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Execution ---
if __name__ == '__main__':
    # Sample grayscale image directory for testing
    base_dir = "./Stylus/final_results/stylized_CFG_temp1_alpha0.9/mel_images"
    save_dir = "./Stylus/asset"

    img_file_list_tmp = glob.glob(f"{base_dir}/*.png")
    img_file_list = []
    for file_tmp in img_file_list_tmp:
        if "_colored" not in file_tmp:
            img_file_list.append(file_tmp)


    for img_file in img_file_list:
        file_name = os.path.split(img_file)[-1].replace(".png", "")
        output_filename = f"{save_dir}/{file_name}_colored.png"
        apply_magma_colormap(img_file, output_filename)
