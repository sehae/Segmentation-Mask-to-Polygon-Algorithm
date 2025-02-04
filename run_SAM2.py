from datetime import datetime
import os
import numpy as np
import cv2 as cv
import torch
import torch_directml
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Setup
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cpu")
print(f"Using device: {device}")

# Load Model
sam2_checkpoint = "C:\\Codes\\Segmentation-Mask-to-Polygon-Algorithm\\SAM2\\sam2_trained.pt"
model_cfg = "C:\\Codes\\Segmentation-Mask-to-Polygon-Algorithm\\SAM2\\sam2.1_hiera_b+.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# Load Image
image = Image.open('Data/sample1.png')
image = np.array(image.convert("RGB"))
masks = mask_generator.generate(image)

# Define folder paths
base_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder = os.path.join(base_dir, "SAM2", "runs", f"sample_{timestamp}")
ground_truth_folder = os.path.join(output_folder, "GroundTruth")
refined_folder = os.path.join(output_folder, "RefinedMasks")
final_folder = os.path.join(output_folder, "FinalOutput")

# Create necessary directories
os.makedirs(output_folder, exist_ok=True)
os.makedirs(ground_truth_folder, exist_ok=True)
os.makedirs(refined_folder, exist_ok=True)
os.makedirs(final_folder, exist_ok=True)

# Save result image
filename = os.path.join(output_folder, "result.png")
Image.fromarray(image).save(filename)

if masks:
    kernel = np.ones((15, 15), np.uint8)
    combined_mask = np.zeros_like(image)

    for i, mask in enumerate(masks):
        try:
            mask_image = mask['segmentation'].astype(np.uint8) * 255
            mask_filename = os.path.join(ground_truth_folder, f"mask_result_{i + 1}.png")
            cv.imwrite(mask_filename, mask_image)

            # Morphological closing
            closed = cv.morphologyEx(mask_image, cv.MORPH_CLOSE, kernel)

            # Flood fill
            flood_filled = closed.copy()
            h, w = closed.shape[:2]
            mask_for_floodfill = np.zeros((h + 2, w + 2), np.uint8)
            cv.floodFill(flood_filled, mask_for_floodfill, seedPoint=(0, 0), newVal=255)
            inverted_flood_filled = cv.bitwise_not(flood_filled)
            refined_mask = closed | inverted_flood_filled

            refined_mask_filename = os.path.join(refined_folder, f"refined_mask_{i + 1}.png")
            cv.imwrite(refined_mask_filename, refined_mask)

            # Contour detection
            gray = cv.cvtColor(cv.imread(refined_mask_filename), cv.COLOR_BGR2GRAY)
            edges = cv.Canny(gray, 30, 200)
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            print(f"Contours found in mask {i + 1}: {len(contours)}")

            final_image = cv.imread(refined_mask_filename)
            cv.drawContours(final_image, contours, -1, (0, 255, 0), 2)

            # Corner detection
            gray_float = np.float32(gray)
            dst = cv.cornerHarris(gray_float, 3, 5, 0.04)
            dst = cv.dilate(dst, None)
            final_image[dst > 0.01 * dst.max()] = [0, 0, 255]

            final_mask_filename = os.path.join(final_folder, f"final_result_{i + 1}.png")
            cv.imwrite(final_mask_filename, final_image)

            # Combine masks
            mask_resized = cv.resize(final_image, (w, h), interpolation=cv.INTER_NEAREST)
            combined_mask = cv.addWeighted(combined_mask, 1.0, mask_resized, 1.0, 0)

        except Exception as e:
            print(f"Error processing mask {i + 1}: {e}")

    combined_mask_path = os.path.join(final_folder, "combined_mask.png")
    cv.imwrite(combined_mask_path, combined_mask)
    print(f"Combined mask saved at: {combined_mask_path}")
else:
    print("No masks detected.")