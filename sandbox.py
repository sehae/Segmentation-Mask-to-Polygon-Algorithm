import cv2 as cv
from ultralytics import YOLO
from datetime import datetime

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch_directml
import numpy as np
import os

import sam2_loader  # Import the module

# Load image
img = cv.imread('Data/sample1.png')

def process_results(results, output_folder, ground_truth_folder, refined_folder, final_folder):
    # Save the result image
    filename = os.path.join(output_folder, "result.png")
    results[0].save(filename=filename)

    # Ground-truth mask
    masks = results[0].masks
    if masks is not None:
        for i, mask in enumerate(masks.data):
            try:
                mask_image = mask.cpu().numpy().astype(np.uint8) * 255
                mask_filename = os.path.join(ground_truth_folder, f"mask_result_{i + 1}.png")
                cv.imwrite(mask_filename, mask_image)
            except Exception as e:
                print(f"Error saving mask {i + 1}: {e}")
    else:
        print("No masks detected.")
        exit(1)

    # Process masks
    kernel = np.ones((15, 15), np.uint8)
    for i, mask in enumerate(masks.data):
        try:
            mask_image = mask.cpu().numpy().astype(np.uint8) * 255
            closed = cv.morphologyEx(mask_image, cv.MORPH_CLOSE, kernel)
            flood_filled = closed.copy()
            h, w = closed.shape[:2]
            mask_for_floodfill = np.zeros((h + 2, w + 2), np.uint8)
            cv.floodFill(flood_filled, mask_for_floodfill, seedPoint=(0, 0), newVal=255)
            inverted_flood_filled = cv.bitwise_not(flood_filled)
            refined_mask = closed | inverted_flood_filled
            mask_filename = os.path.join(refined_folder, f"refined_mask_{i + 1}.png")
            cv.imwrite(mask_filename, refined_mask)
        except Exception as e:
            print(f"Error processing mask {i + 1}: {e}")

    # Contour and corner detection
    for i, mask in enumerate(masks.data):
        try:
            mask_path = os.path.join(refined_folder, f"refined_mask_{i + 1}.png")
            image = cv.imread(mask_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            edge = cv.Canny(gray, 30, 200)
            contours, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(image, contours, -1, (0, 255, 0), 2)
            gray = np.float32(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
            dst = cv.cornerHarris(gray, 3, 5, 0.04)
            dst = cv.dilate(dst, None)
            image[dst > 0.01 * dst.max()] = [0, 0, 255]
            final_name = os.path.join(final_folder, f"final_result_{i + 1}.png")
            cv.imwrite(final_name, image)
        except Exception as e:
            print(f"Error processing contours for mask {i + 1}: {e}")

    # Combine masks
    h, w, _ = img.shape
    combined_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, mask in enumerate(masks.data):
        try:
            mask_path = os.path.join(final_folder, f"final_result_{i + 1}.png")
            mask_image = cv.imread(mask_path, cv.IMREAD_COLOR)
            mask_image = cv.resize(mask_image, (w, h), interpolation=cv.INTER_NEAREST)
            combined_mask = cv.addWeighted(combined_mask, 1.0, mask_image, 1.0, 0)
        except Exception as e:
            print(f"Error combining mask {i + 1}: {e}")

    combined_mask_path = os.path.join(final_folder, "combined_mask.png")
    cv.imwrite(combined_mask_path, combined_mask)
    print(f"Combined mask saved at: {combined_mask_path}")

def main():
    while True:
        print("\nMenu:")
        print("1. Load SAM2 Model")
        print("2. Load YOLOv11 Model")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            device = torch.device("cpu")
            print(f"using device: {device}")
            sam2_checkpoint = "C:\\Codes\\Segmentation-Mask-to-Polygon-Algorithm\\SAM2\\sam2_trained.pt"
            model_cfg = "C:\\Codes\\Segmentation-Mask-to-Polygon-Algorithm\\SAM2\\sam2.1_hiera_b+.yaml"
            sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
            mask_generator = SAM2AutomaticMaskGenerator(sam2)
            results = mask_generator.generate(img)
            process_results(results, "output", "ground_truth", "refined", "final")
        elif choice == "2":
            print("Loading YOLOv11 model...")
            model = YOLO("YOLOv11 Instance Segmentation/yolov11_instance_trained.pt")
            results = model(img)
            print("Model loaded.")
            process_results(results, "output", "ground_truth", "refined", "final")
        elif choice == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter between 1 and 3.")

if __name__ == "__main__":
    main()
