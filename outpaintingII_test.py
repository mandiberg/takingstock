from pipeline_stable_diffusion_xl_differential_img2img import StableDiffusionXLDifferentialImg2ImgPipeline

import random
import urllib.request

import cv2
import numpy as np
import torch

torch.set_default_device("mps")

# from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline


model_id="SG161222/RealVisXL_V4.0"
custom_pipeline_path = "pipeline_stable_diffusion_xl_differential_img2img"
pipeline = StableDiffusionXLDifferentialImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
).to("mps")


# pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "SG161222/RealVisXL_V4.0",
#     torch_dtype=torch.float16,
#     variant="fp16",
# ).to("mps")
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

# pipeline = StableDiffusionXLPipeline.from_pretrained(
#     "SG161222/RealVisXL_V4.0",
#     torch_dtype=torch.float16,
#     variant="fp16",
#     custom_pipeline="pipeline_stable_diffusion_xl_differential_img2img",
# ).to("cuda")
# pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)

pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name=[
        "ip-adapter-plus_sdxl_vit-h.safetensors",
    ],
    image_encoder_folder="models/image_encoder",
)
pipeline.set_ip_adapter_scale(0.1)


def merge_images(original, new_image, offset, direction):
    if direction in ["left", "right"]:
        merged_image = np.zeros((original.shape[0], original.shape[1] + offset, 3), dtype=np.uint8)
    elif direction in ["top", "bottom"]:
        merged_image = np.zeros((original.shape[0] + offset, original.shape[1], 3), dtype=np.uint8)

    if direction == "left":
        merged_image[:, offset:] = original
        merged_image[:, : new_image.shape[1]] = new_image
    elif direction == "right":
        merged_image[:, : original.shape[1]] = original
        merged_image[:, original.shape[1] + offset - new_image.shape[1] : original.shape[1] + offset] = new_image
    elif direction == "top":
        merged_image[offset:, :] = original
        merged_image[: new_image.shape[0], :] = new_image
    elif direction == "bottom":
        merged_image[: original.shape[0], :] = original
        merged_image[original.shape[0] + offset - new_image.shape[0] : original.shape[0] + offset, :] = new_image

    return merged_image


def slice_image(image):
    height, width, _ = image.shape
    slice_size = min(width // 2, height // 3)

    slices = []

    for h in range(3):
        for w in range(2):
            left = w * slice_size
            upper = h * slice_size
            right = left + slice_size
            lower = upper + slice_size

            if w == 1 and right > width:
                left -= right - width
                right = width
            if h == 2 and lower > height:
                upper -= lower - height
                lower = height

            slice = image[upper:lower, left:right]
            slices.append(slice)

    return slices


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




def generate_image(prompt, negative_prompt, image, mask, ip_adapter_image, seed: int = None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    print(type(image))
    print(type(mask))
    print(type(ip_adapter_image))


    # generator = torch.Generator(device="mps").manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        guidance_scale=4.0,
        num_inference_steps=25,
        original_image=image,
        image=image,
        strength=1.0,
        map=mask,
        generator=generator,
        ip_adapter_image=[ip_adapter_image],
        output_type="np",
    ).images[0]
    
    # .cpu().numpy()

    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


prompt = "Beauty fashion portrait of young blond woman model with natural makeup and perfect skin with bright Crimson chrysanthemum flower posing in s"
negative_prompt = ""
direction = "top"  # left, right, top, bottom
inpaint_mask_color = 50  # lighter use more of the Telea inpainting
expand_pixels = 256  # I recommend to don't go more than half of the picture so it has context
times_to_expand = 1 # this was set to 4. dropping to 1 reduced time to 40% of original because only 2 rounds, not 5.

# url = "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/photo-1711580377289-eecd23d00370.jpeg?download=true"
# with urllib.request.urlopen(url) as url_response:
#     img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
# original = cv2.imdecode(img_array, -1)

path = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images/images_shutterstock/C/CC/1046510584.jpg"
# path = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images/images_getty/D/D6/peruvian-woman-weaving-carpet-sacred-valley-picture-id516136941.jpg"
# path = "/Users/michaelmandiberg/Documents/projects-active/facemap_production/segment_images/images_adobe/0/01/191746182.jpg"

img_array = cv2.imread(path)
# If you want to ensure the array is of type np.uint8, you can use astype
img_array = img_array.astype(np.uint8)
original = img_array

image = image_resize(original)
expand_pixels_to_square = 1024 - image.shape[1]  # image.shape[1] for horizontal, image.shape[0] for vertical
image, mask = process_image(
    image, expand_pixels=expand_pixels_to_square, direction=direction, inpaint_mask_color=inpaint_mask_color
)

ip_adapter_image = []
for index, part in enumerate(slice_image(original)):
    ip_adapter_image.append(part)

generated = generate_image(prompt, negative_prompt, image, mask, ip_adapter_image)
final_image = generated

for i in range(times_to_expand):
    image, mask = process_image(
        final_image, direction=direction, expand_pixels=expand_pixels, inpaint_mask_color=inpaint_mask_color
    )

    ip_adapter_image = []
    for index, part in enumerate(slice_image(generated)):
        ip_adapter_image.append(part)

    generated = generate_image(prompt, negative_prompt, image, mask, ip_adapter_image)
    final_image = merge_images(final_image, generated, 256, direction)

cv2.imwrite("result.png", final_image)
