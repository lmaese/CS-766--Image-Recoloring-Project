from models import GeneratorRRDB
from datasets import denormalize, mean, std
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, required=True, help="Path to image")
parser.add_argument("--checkpoint_model", type=str, required=True, help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--residual_blocks", type=int, default=23, help="Number of residual blocks in G")
opt = parser.parse_args()
print(opt)

os.makedirs("images/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

tensor = transforms.ToTensor()

transform = transforms.Compose([transforms.Resize((218 // 4, 218 // 4), Image.BICUBIC),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std),
                                ])

lr_transform = transforms.Compose([transforms.Resize((218 // 4, 218 // 4), Image.BICUBIC),
                                transforms.Grayscale(3),
                                transforms.ToTensor(),
                                ])

untransform = transforms.Resize((218, 178))

hr_image = Image.open(opt.image_path)

lr_image = transform(hr_image)

# Prepare input
image_tensor = Variable(lr_image).to(device).unsqueeze(0)

# Upsample image
with torch.no_grad():
    sr_image = denormalize(generator(image_tensor)).cpu()

import torch.nn as nn

# Save image
fn = opt.image_path.split("/")[-1][:-4]
save_image(untransform(sr_image), f"images/outputs/sr-{fn}.png")
save_image(untransform(lr_transform(hr_image)), f"images/outputs/lr-{fn}.png")
#save_image(untransform(denormalize(nn.functional.interpolate(lr_image, scale_factor=4))), f"images/outputs/lr-{fn}.png")
save_image(tensor(hr_image), f"images/outputs/hr-{fn}.png")


