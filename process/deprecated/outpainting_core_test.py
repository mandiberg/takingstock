import random
import urllib.request

import cv2
import numpy as np
import torch

torch.set_default_device("mps")

# from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

def image_resize(image, new_size=1024):
    height, width = image.shape[:2]

    aspect_ratio = width / height
    new_width = new_size
    new_height = new_size

    if aspect_ratio != 1:
        if width > height:
            new_height = int(new_size / aspect_ratio)
        else:
            new_width = int(new_size * aspect_ratio)

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return image

def process_image(
    image,
    fill_color=(0, 0, 0),
    mask_offset=50,
    blur_radius=500,
    expand_pixels=256,
    direction="left",
    inpaint_mask_color=50,
    max_size=1024,
):
    height, width = image.shape[:2]

    new_height = height + (expand_pixels if direction in ["top", "bottom"] else 0)
    new_width = width + (expand_pixels if direction in ["left", "right"] else 0)

    if new_height > max_size:
        # If so, crop the image from the opposite side
        if direction == "top":
            image = image[:max_size, :]
        elif direction == "bottom":
            image = image[new_height - max_size :, :]
        new_height = max_size

    if new_width > max_size:
        # If so, crop the image from the opposite side
        if direction == "left":
            image = image[:, :max_size]
        elif direction == "right":
            image = image[:, new_width - max_size :]
        new_width = max_size

    height, width = image.shape[:2]

    new_image = np.full((new_height, new_width, 3), fill_color, dtype=np.uint8)
    mask = np.full_like(new_image, 255, dtype=np.uint8)
    inpaint_mask = np.full_like(new_image, 0, dtype=np.uint8)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    inpaint_mask = cv2.cvtColor(inpaint_mask, cv2.COLOR_BGR2GRAY)

    if direction == "left":
        new_image[:, expand_pixels:] = image[:, : max_size - expand_pixels]
        mask[:, : expand_pixels + mask_offset] = inpaint_mask_color
        inpaint_mask[:, :expand_pixels] = 255
    elif direction == "right":
        new_image[:, :width] = image
        mask[:, width - mask_offset :] = inpaint_mask_color
        inpaint_mask[:, width:] = 255
    elif direction == "top":
        new_image[expand_pixels:, :] = image[: max_size - expand_pixels, :]
        mask[: expand_pixels + mask_offset, :] = inpaint_mask_color
        inpaint_mask[:expand_pixels, :] = 255
    elif direction == "bottom":
        new_image[:height, :] = image
        mask[height - mask_offset :, :] = inpaint_mask_color
        inpaint_mask[height:, :] = 255

    # mask blur
    if blur_radius % 2 == 0:
        blur_radius += 1
    mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

    # telea inpaint
    _, mask_np = cv2.threshold(inpaint_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inpaint = cv2.inpaint(new_image, mask_np, 3, cv2.INPAINT_TELEA)

    # convert image to tensor
    inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
    inpaint = torch.from_numpy(inpaint).permute(2, 0, 1).float()
    inpaint = inpaint / 127.5 - 1
    inpaint = inpaint.unsqueeze(0).to("mps")

    # convert mask to tensor
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0).float() / 255.0
    mask = mask.to("mps")

    return inpaint, mask



prompt = ""
negative_prompt = ""
direction = "right"  # left, right, top, bottom
inpaint_mask_color = 50  # lighter use more of the Telea inpainting
expand_pixels = 256  # I recommend to don't go more than half of the picture so it has context
times_to_expand = 4


url = "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/photo-1711580377289-eecd23d00370.jpeg?download=true"

with urllib.request.urlopen(url) as url_response:
    img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)

original = cv2.imdecode(img_array, -1)
image = image_resize(original)
expand_pixels_to_square = 1024 - image.shape[1]  # image.shape[1] for horizontal, image.shape[0] for vertical
image, mask = process_image(
    image, expand_pixels=expand_pixels_to_square, direction=direction, inpaint_mask_color=inpaint_mask_color
)


pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("mps")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

# Print image and mask shapes
print(image.shape)
print(mask.shape)

# Convert tensor to numpy with cpu
image_np = image.cpu().numpy()
mask_np = mask.cpu().numpy()

# Print image and mask sizes
print(image_np.shape)
print(mask_np.shape)

# Convert numpy arrays back to PyTorch tensors on the MPS device
image = torch.tensor(image_np, device="mps", dtype=torch.float16)
mask = torch.tensor(mask_np, device="mps", dtype=torch.float16)

# Ensure the shapes and types are correct
print(image.shape)
print(mask.shape)

# Run the pipeline
image = pipeline(
    prompt="",
    negative_prompt="",
    width=1024,
    height=1024,
    guidance_scale=6.0,
    num_inference_steps=25,
    original_image=image,
    image=image,
    strength=1.0,
    map=mask
).images[0]

# Convert the result to a numpy array and process it
result = (image * 255).cpu().numpy().astype(np.uint8)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

# Print the final image shape
print(result.shape)

cv2.imshow("image", image)
