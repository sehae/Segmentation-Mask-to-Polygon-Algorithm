import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import cv2

# Load Trained Model
model = YOLO("YOLOv11 Instance Segmentation/best.pt")

# Run Inference
results = model("Data/sample1.png")

# Get timestamp for unique folder naming
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define folder paths
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
output_folder = os.path.join(base_dir, "YOLOv11 Instance Segmentation", "runs", f"sample_{timestamp}")
contour_folder = os.path.join(output_folder, "Contour")
final_folder = os.path.join(output_folder, "Final Output")

# Create necessary directories
try:
    os.makedirs(output_folder, exist_ok=True)
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
    for i, mask in enumerate(masks.data):
        try:
            mask_image = mask.cpu().numpy().astype(np.uint8) * 255
            mask_filename = os.path.join(output_folder, f"mask_result_{i + 1}.png")
            cv2.imwrite(mask_filename, mask_image)
        except Exception as e:
            print(f"Error saving mask {i + 1}: {e}")
else:
    print("No masks detected.")
    exit(1)

# Contour Detection
for i, mask in enumerate(masks.data):
    try:
        mask_image_path = os.path.join(output_folder, f"mask_result_{i + 1}.png")
        image = cv2.imread(mask_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(f"Number of contours found in mask {i + 1}: {len(contours)}")

        # Draw contours on the image
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        contour_filename = os.path.join(contour_folder, f"contour_result_{i + 1}.png")
        cv2.imwrite(contour_filename, image)
    except Exception as e:
        print(f"Error processing contours for mask {i + 1}: {e}")

# Corner Detection
for i, mask in enumerate(masks.data):
    try:
        contour_image_path = os.path.join(contour_folder, f"contour_result_{i + 1}.png")
        image = cv2.imread(contour_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # Dilate to enhance corners for visibility
        dst = cv2.dilate(dst, None)

        # Threshold for optimal corner detection
        image[dst > 0.01 * dst.max()] = [0, 0, 255]
        final_filename = os.path.join(final_folder, f"final_result_{i + 1}.png")
        cv2.imwrite(final_filename, image)
    except Exception as e:
        print(f"Error detecting corners for contour {i + 1}: {e}")
