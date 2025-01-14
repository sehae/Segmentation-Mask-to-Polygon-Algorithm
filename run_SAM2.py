from datetime import datetime

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch_directml
import numpy as np
import os

#Setup
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                  sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(
                contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)

print(os.getcwd())

amd_gpu = torch_directml.device()

#Load Model
sam2_checkpoint = "C:\\Codes\\Segmentation-Mask-to-Polygon-Algorithm\\SAM2\\checkpoint.pt"
model_cfg = "C:\\Codes\\Segmentation-Mask-to-Polygon-Algorithm\\SAM2\\sam2.1_hiera_b+.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

#Data Feeding
image = Image.open('Data/sample2.jpg')
image = np.array(image.convert("RGB"))
masks = mask_generator.generate(image)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#Define folder paths
base_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(base_dir, "SAM2", "runs", f"sample_{timestamp}")
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

#Save the result image
filename = os.path.join(output_folder, "result.png")
image[0].save(filename=filename)