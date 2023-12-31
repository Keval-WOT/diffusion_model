from diffusers import StableDiffusionImageVariationPipeline
import torch
import os 
from tqdm import tqdm as tqdm

from PIL import Image
import cv2
from torchvision import transforms
def diffusion_model(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == 'cuda': #cuda Sync
        torch.cuda.synchronize()

        
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
            [0.26862954, 0.26130258, 0.27577711]),])
    inp = tform(im).to(device)
    print("Input Shape",inp.size())

    out = sd_pipe(inp, guidance_scale=3)
    print(type(out["images"][0].save("result.jpg")))
    


if __name__=='__main__':
    input_path = '/kaggle/working/diffusion_model/results'
    output_path='/kaggle/working/diffusion_model'
    if os.path.exists(os.path.join(output_path,'Result_images')):
        os.mkidr(os.path.join(output_path,'Result_images'))
    for image in os.listdir(input_path):
        result = diffusion_model(os.path.join(input_path,image))
        