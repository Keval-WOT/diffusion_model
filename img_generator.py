from diffusers import StableDiffusionImageVariationPipeline
import torch
import os 
from tqdm import tqdm as tqdm
import gradio
from PIL import Image
import cv2
from torchvision import transforms
def diffusion_model(image_path):
    image = cv2.imread(image_path)
    
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    revision="v2.0",
    )
    sd_pipe = sd_pipe.to(device)

    im = Image.open(image_path)
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
        ),
        transforms.Normalize(
            [0.48145466, 0.4578275, 0.40821073],
            [0.26862954, 0.26130258, 0.27577711]),
])
    inp = tform(im).to(device)

    out = sd_pipe(inp, guidance_scale=3)
    out["images"][0].save("result.jpg")



if __name__=='__main__':
    input_path = input("Enter Input Path")
    output_path=input('Enter Output path')
    if os.path.exists(os.join(output_path,'Result_images')):
        os.mkidr(os.path.join(output_path,'Result_images'))
    for image in os.listdir(input_path):
        result = diffusion_model(os.path.join(input_path,image))
        