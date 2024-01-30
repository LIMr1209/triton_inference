from src.u2net.detect import load_model, preprocess
import numpy as np
from PIL import Image
import torch

model = load_model()

image_path = "example.jpg"

img = Image.open(image_path).convert('RGB')


sample = preprocess(np.array(img))

ipt = sample["image"].unsqueeze(0).float()
script_model = torch.jit.trace(model, ipt, strict=True)
torch.jit.save(script_model, "../torchscript_background/1/model.pt")
