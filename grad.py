import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import sys

if len(sys.argv) != 3:
    print("Usage: python grad.py <model_path> <image_path>")
    sys.exit(1)

image_path = sys.argv[2]
if not image_path.lower().endswith((".png", ".jpg", ".jpeg")):
    print("Please provide a valid image file path.")
    sys.exit(1)

model_path = sys.argv[1]
if not model_path.lower().endswith(".pth"):
    print("Please provide a valid model file path with .pth extension.")
    sys.exit(1)

# 1. Load the model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


# 2. Preprocess the image
raw_image = Image.open(image_path).convert("RGB")
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
input_tensor = transform(raw_image).unsqueeze(0)  # shape: (1, 3, 224, 224)

# 3. Grad-CAM setup
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# 4. Get CAM
grayscale_cam = cam(input_tensor=input_tensor)[0, :]  # shape: (224, 224)

# 5. Prepare original image for overlay (de-normalize)
img_np = np.array(raw_image.resize((224, 224))).astype(np.float32) / 255.0

# 6. Overlay CAM on image
cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

plt.imshow(cam_image)
plt.axis("off")
plt.title("Grad-CAM Visualization")
plt.show()
