from outpainting_modular import outpaint, image_resize

import urllib.request
import cv2
import numpy as np
import torch


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