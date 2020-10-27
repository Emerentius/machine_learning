# book: Deep Learning with PyTorch

#%%
from torchvision import models
from typing import List

import torch

# %%
import json

# Torchvision's pretrained models don't include any labels for their output, so you need to find and download those on your own. Ridiculous, really.
# Downloaded labels from https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
# and then modified it to be actually loadable.
with open("imagenet_1000_class_labels.json") as f:
    labels = json.load(f)

# %%
resnet = models.resnet101(pretrained=True)
resnet.eval()
# %%
from torchvision import transforms

img_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
)
preprocess = transforms.Compose(
    [
        img_transforms,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# %%
def full_preprocess(image):
    img_tensor = preprocess(image)
    batch_tensor = torch.unsqueeze(img_tensor, 0)
    return batch_tensor


# %%
from typing import Tuple


def most_likely_label(image) -> Tuple[str, float]:
    out = resnet(full_preprocess(image))

    _max_vals, indices = torch.max(out, 1)
    index = indices[0]
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    # labels[index[0]], percentage[index[0]]
    return labels[index], percentage[index].item()


# %%


from PIL import Image

# img = Image.open("lisa.jpeg")
# img2 = Image.open("lisa_simple.jpeg")

# cat pics (not part of the repo)
# images = ["lisa.jpeg", "lisa_simple.jpeg"]

images: List[str] = []

for image in images:
    img = Image.open(image)
    print(most_likely_label(img))
# %%
