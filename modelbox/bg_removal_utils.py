import numpy as np
from PIL import Image


def prepare_input_image(pil_image):
    """Prepare input image for the BRIA background removal model.
    
    Args:
        pil_image: PIL Image to process
        
    Returns:
        Tuple of (original PIL image, preprocessed numpy array)
    """
    input_img = pil_image.convert("RGB")
    # Resize to model input size
    input_img_np = np.array(
        input_img.resize((1024, 1024), resample=Image.Resampling.BILINEAR)
    )
    # Normalize
    input_img_np = input_img_np / 255.0
    # Normalize with mean 0.5 and std 1.0
    input_img_np = (input_img_np - 0.5) / 0.5
    # Change to NCHW format
    input_img_np = input_img_np.transpose(2, 0, 1)
    input_img_np = input_img_np[None, ...].astype(np.float32)
    return input_img, input_img_np


def post_process_mask(mask_np, input_img):
    """Process the output mask from the BRIA background removal model.
    
    Args:
        mask_np: Numpy array containing the mask from model output
        input_img: Original PIL image
        
    Returns:
        PIL Image with transparency applied based on the mask
    """
    # Resize mask to original image size
    mask_img = Image.fromarray(mask_np)
    mask_img = mask_img.resize(
        (input_img.width, input_img.height), resample=Image.Resampling.BILINEAR
    )
    
    # Convert mask to proper alpha channel format (0-255)
    mask_np = np.array(mask_img)
    # Normalize to 0-255 range
    if mask_np.max() > 0:  # Avoid division by zero
        mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-6) * 255
    mask_np = mask_np.astype(np.uint8)
    
    # Apply mask as alpha channel
    result_img = input_img.copy()
    result_img.putalpha(Image.fromarray(mask_np))
    
    return result_img