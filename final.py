import os
from datetime import datetime

import cv2 as cv
import numpy as np
from ultralytics import YOLO

# Load image
img = cv.imread('Data/sample1.png')

# Load Trained Model
model = YOLO("YOLOv11 Instance Segmentation/yolov11_instance_trained.pt")

# Run Inference
results = model(img)

# Get timestamp for unique folder naming
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define folder paths
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
output_folder = os.path.join(base_dir, "YOLOv11 Instance Segmentation", "runs", f"sample_{timestamp}")
refined_folder = os.path.join(output_folder, "Refined Masks")
contour_folder = os.path.join(output_folder, "Contour")
final_folder = os.path.join(output_folder, "Final Output")

# Create necessary directories
try:
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(refined_folder, exist_ok=True)
    os.makedirs(contour_folder, exist_ok=True)
    os.makedirs(final_folder, exist_ok=True)
except Exception as e:
    print(f"Error creating folders: {e}")
    exit(1)

# Save the result image
filename = os.path.join(output_folder, "result.png")
results[0].save(filename=filename)

# Process masks
masks = results[0].masks
if masks is not None:
    kernel = np.ones((15, 15), np.uint8)
    for i, mask in enumerate(masks.data):
        try:
            # Convert YOLO mask to a usable format
            mask_image = mask.cpu().numpy().astype(np.uint8) * 255

            # Morphological closing to close small holes
            closed = cv.morphologyEx(mask_image, cv.MORPH_CLOSE, kernel)

            # Flood fill to fill internal gaps
            flood_filled = closed.copy()
            h, w = closed.shape[:2]
            mask_for_floodfill = np.zeros((h + 2, w + 2), np.uint8)  # Padding for floodFill
            cv.floodFill(flood_filled, mask_for_floodfill, seedPoint=(0, 0), newVal=255)
            inverted_flood_filled = cv.bitwise_not(flood_filled)

            # Combine closed mask with flood-filled result
            refined_mask = closed | inverted_flood_filled

            # Save the refined mask
            mask_filename = os.path.join(refined_folder, f"refined_mask_{i + 1}.png")
            cv.imwrite(mask_filename, refined_mask)
        except Exception as e:
            print(f"Error processing mask {i + 1}: {e}")
else:
    print("No masks detected.")
    exit(1)

if masks is not None:
    for i, mask in enumerate(masks.data):
        try:
            # Convert mask to grayscale
            mask_path = os.path.join(refined_folder, f"refined_mask_{i + 1}.png")
            image = cv.imread(mask_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Apply Canny edge detection
            edge = cv.Canny(gray, 30, 200)
            contours, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            print(f"Number of contours found in mask {i + 1}: {len(contours)}")

            # Draw contours on the image
            cv.drawContours(image, contours, -1, (0, 255, 0), 2)

            # Corner Detection
            gray = np.float32(cv.cvtColor(image,cv.COLOR_BGR2GRAY))
            dst = cv.cornerHarris(gray,3,5,0.04)
            dst = cv.dilate(dst,None)

            # Highlight corners in red
            image[dst > 0.01 * dst.max()] = [0, 0, 255]

            # Save the final image
            final_name = os.path.join(final_folder, f"final_result_{i + 1}.png")
            cv.imwrite(final_name, image)

        except Exception as e:
            print(f"Error processing contours for mask {i + 1}: {e}")

