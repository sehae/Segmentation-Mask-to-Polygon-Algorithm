import cv2 as cv
import numpy as np
import os


# Function to calculate Intersection over Union (IoU)
def calculate_iou(mask_pred, mask_gt):
    intersection = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum()
    return intersection / union if union > 0 else 0


# Define paths
yolo_output_dir = "YOLOv11 Instance Segmentation/runs/sample_2025-02-01_19-29-17/"
refined_mask_dir = os.path.join(yolo_output_dir, "Refined Masks")
final_output_dir = os.path.join(yolo_output_dir, "Final Output")
ground_truth_dir = os.path.join(yolo_output_dir, "Ground Truth")

# Number of test images
num_images = len(os.listdir(ground_truth_dir))

# Store IoU results
yolo_ious, refined_ious, final_ious = [], [], []

# Iterate over test images
for i in range(1, num_images + 1):
    # Load ground truth mask
    gt_mask_path = os.path.join(ground_truth_dir, f"mask_result_{i + 1}.png")
    if os.path.exists(gt_mask_path):
        gt_mask = cv.imread(gt_mask_path, cv.IMREAD_GRAYSCALE)
        if gt_mask is not None:
            gt_mask = (gt_mask > 128).astype(np.uint8)
        else:
            print(f"Error: Could not read ground truth mask: {gt_mask_path}")
            continue
    else:
        print(f"Error: Ground truth mask not found: {gt_mask_path}")
        continue

    # Load YOLO mask
    yolo_mask_path = os.path.join(yolo_output_dir, f"result.png")  # Adjust filename if needed
    yolo_mask = cv.imread(yolo_mask_path, cv.IMREAD_GRAYSCALE)
    if yolo_mask is not None:
        yolo_mask = (yolo_mask > 128).astype(np.uint8)
    else:
        print(f"Error: Could not read YOLO mask: {yolo_mask_path}")
        continue

    # Resize the masks to the same shape (using ground truth mask size as reference)
    yolo_mask_resized = cv.resize(yolo_mask, (gt_mask.shape[1], gt_mask.shape[0]))

    # Load refined mask
    refined_mask_path = os.path.join(refined_mask_dir, f"refined_mask_{i}.png")
    refined_mask = cv.imread(refined_mask_path, cv.IMREAD_GRAYSCALE)
    if refined_mask is not None:
        refined_mask = (refined_mask > 128).astype(np.uint8)
        # Resize the refined mask
        refined_mask = cv.resize(refined_mask, (gt_mask.shape[1], gt_mask.shape[0]))
    else:
        print(f"Error: Could not read refined mask: {refined_mask_path}")
        continue

    # Load final output mask
    final_mask_path = os.path.join(final_output_dir, f"final_result_{i}.png")
    final_mask = cv.imread(final_mask_path, cv.IMREAD_GRAYSCALE)
    if final_mask is not None:
        final_mask = (final_mask > 128).astype(np.uint8)
        # Resize the final mask
        final_mask = cv.resize(final_mask, (gt_mask.shape[1], gt_mask.shape[0]))
    else:
        print(f"Error: Could not read final mask: {final_mask_path}")
        continue

    # Compute IoU for each stage
    iou_yolo = calculate_iou(yolo_mask_resized, gt_mask)
    iou_refined = calculate_iou(refined_mask, gt_mask)
    iou_final = calculate_iou(final_mask, gt_mask)

    # Store results
    yolo_ious.append(iou_yolo)
    refined_ious.append(iou_refined)
    final_ious.append(iou_final)

    # Print results for each image
    print(f"\n[Image {i}] IoU Scores:")
    print(f"   YOLO Mask IoU: {iou_yolo:.4f}")
    print(f"   Refined Mask IoU: {iou_refined:.4f}")
    print(f"   Final Processed Mask IoU: {iou_final:.4f}")

# Compute Average IoU
avg_yolo_iou = np.mean(yolo_ious)
avg_refined_iou = np.mean(refined_ious)
avg_final_iou = np.mean(final_ious)

# Print Summary
print("\n===== OVERALL MASK EVALUATION =====")
print(f"📌 Average YOLO IoU: {avg_yolo_iou:.4f}")
print(f"📌 Average Refined Mask IoU: {avg_refined_iou:.4f}")
print(f"📌 Average Final Processed Mask IoU: {avg_final_iou:.4f}")
