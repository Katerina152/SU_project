import torch
from PIL import Image
from transformers import ViTModel, AutoImageProcessor
from pathlib import Path

# 1. Load model + processor
model_name = "google/vit-base-patch16-224"
vit = ViTModel.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

vit.eval()

# 2. Load an image from the data folder
image_path = Path(__file__).parent.parent / "data" / "ISIC2019"/ "train"/ "images"/ "ISIC_2019_Training_Input"/ "ISIC_0000360.jpg"
image = Image.open(image_path).convert("RGB")
#/Users/katerinaskrika/Desktop/cool_project/data/ISIC2019/train/images/ISIC_2019_Training_Input/ISIC_0000360.jpg


# 3. Convert image -> tensor using processor
inputs = processor(images=image, return_tensors="pt")

# inputs now contains:
# inputs["pixel_values"] -> shape [1, 3, 224, 224]

# 4. Forward pass through ViT
with torch.no_grad():
    outputs = vit(**inputs)

# 5. Extract embeddings
last_hidden = outputs.last_hidden_state      # [1, seq_len, 768]
cls_embedding = last_hidden[:, 0]           # CLS token embedding

print("CLS embedding shape:", cls_embedding.shape)

