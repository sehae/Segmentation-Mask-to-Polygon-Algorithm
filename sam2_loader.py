from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt
import torch
import torch_directml
import numpy as np
import os

# Setup
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

def load_model():
    """Loads the SAM2 model and mask generator"""
    print("Loading SAM2 model...")

    sam2_checkpoint = "C:\\Codes\\Segmentation-Mask-to-Polygon-Algorithm\\SAM2\\sam2_trained.pt"
    model_cfg = "C:\\Codes\\Segmentation-Mask-to-Polygon-Algorithm\\SAM2\\sam2.1_hiera_b+.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)

    print("Model loaded successfully!")
    return sam2, mask_generator  # Return the model and generator for use

