from pipeline_stable_diffusion_xl_differential_img2img import StableDiffusionXLDifferentialImg2ImgPipeline
import random
import urllib.request
import cv2
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline



def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA device detected. Using CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device detected. Using MPS.")
    else:
        device = torch.device("cpu")
        print("No CUDA or MPS device detected. Using CPU.")
    return device

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
    
def prepare_mask(image,extension_pixels,inpaint_mask_color=50,mask_offset=50,VERBOSE=True):
    if VERBOSE:print("starting mask preparation")
    height, width = image.shape[:2]
    top, bottom, left, right = extension_pixels["top"], extension_pixels["bottom"], extension_pixels["left"],extension_pixels["right"]
    extended_img = np.zeros((height + top+bottom, width+left+right, 3), dtype=np.uint8)
    extended_img[top:height+top, left:width+left,:] = image

    inpaint_mask = np.zeros_like(extended_img[:, :, 0])
    inpaint_mask[:top+mask_offset,:] = 255
    inpaint_mask[:,:left+mask_offset] = 255
    inpaint_mask[(height+top-mask_offset):,:] = 255
    inpaint_mask[:,(width+left-mask_offset):] = 255

    mask = np.zeros_like(extended_img[:, :, 0])+255
    mask[:top,:] = inpaint_mask_color
    mask[:,:left] = inpaint_mask_color
    mask[(height+top):,:] = inpaint_mask_color
    mask[:,(width+left):] = inpaint_mask_color
    if VERBOSE:print("mask preparation done")
    return extended_img,mask,inpaint_mask


def process_image(extended_img,mask,inpaint_mask,downsampling_scale,blur_radius=500):
    # mask blur
    if blur_radius % 2 == 0:blur_radius += 1
    mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

    # telea inpaint
    _, mask_np = cv2.threshold(inpaint_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    extended_img = cv2.inpaint(extended_img, mask_np, 3, cv2.INPAINT_TELEA)

    n_height,n_width=extended_img.shape[:2]
    extended_img = cv2.resize(extended_img, (n_width//downsampling_scale, n_height//downsampling_scale), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (n_width//downsampling_scale, n_height//downsampling_scale), interpolation = cv2.INTER_AREA)
    return extended_img, mask

def np_to_tensor(inpaint, mask):
    # convert image to tensor
    inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
    inpaint = torch.from_numpy(inpaint).permute(2, 0, 1).float()
    inpaint = inpaint / 127.5 - 1
    inpaint = inpaint.unsqueeze(0).to(device)

    # convert mask to tensor
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0).float() / 255.0
    mask = mask.to(device)
    return inpaint, mask

def merge_image(img,outpaint,extension_pixels,offset=50):
    #### use inpainting for the extended part, but use original for non extend to keep image sharp ###
    outpaint[extension_pixels["top"]+offset:extension_pixels["top"]-offset+np.shape(img)[0],extension_pixels["left"]+offset:extension_pixels["left"]-offset+np.shape(img)[1]]=img
    return outpaint

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

def generate_image(prompt, negative_prompt, image, mask, ip_adapter_image, seed: int = None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=4.0,
        num_inference_steps=12,
        original_image=image,
        image=image,
        strength=1.0,
        map=mask,
        generator=generator,
        ip_adapter_image=[ip_adapter_image],
        output_type="np",
    ).images[0]

    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def outpaint(image,extension_pixels,downsampling_scale=4,prompt="",negative_prompt=""):
    extended_img,mask,inpaint_mask=prepare_mask(image,extension_pixels)
    n_height,n_width=extended_img.shape[:2]
    extended_img, mask = process_image(extended_img,mask,inpaint_mask,downsampling_scale)
    ip_adapter_image = []
    for index, part in enumerate(slice_image(image)):
        ip_adapter_image.append(part)
    extended_img, mask=np_to_tensor(extended_img, mask)
    generated = generate_image(prompt, negative_prompt, extended_img, mask, ip_adapter_image)
    generated = cv2.resize(generated, (n_width,n_height), interpolation = cv2.INTER_LANCZOS4)
    outpaint_image = merge_image(image,generated,extension_pixels)
    return outpaint_image

device = get_device()

# model_id="stabilityai/stable-diffusion-xl-base-1.0"
model_id="SG161222/RealVisXL_V4.0"
pipeline = StableDiffusionXLDifferentialImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
).to(device)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name=[
        "ip-adapter-plus_sdxl_vit-h.safetensors",
    ],
    image_encoder_folder="models/image_encoder",
)
pipeline.set_ip_adapter_scale(0.1)

def main():
    torch.cuda.empty_cache()
    prompt = ""
    negative_prompt = ""
    times_to_expand = 2
    extension_pixels={"top":200,"right":200,"bottom":200,"left":200}

    url = "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/differential/photo-1711580377289-eecd23d00370.jpeg?download=true"

    with urllib.request.urlopen(url) as url_response:
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)

    original = cv2.imdecode(img_array, -1)
    final_image=image_resize(original)
    for i in range(times_to_expand):
        final_image=outpaint(final_image,extension_pixels,downsampling_scale=1,prompt="",negative_prompt="")

    cv2.imwrite("result.png", final_image)    


if __name__=="__main__": 
    main()