import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import sys


def load_model(checkpoint_path):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 1)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, 224, 224)


def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        label = "Tumor" if prob > 0.5 else "No Tumor"
        print(f"Prediction: {label} (probability: {prob:.3f})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <model_path> <image_path>")
        exit(1)
    model_path, image_path = sys.argv[1], sys.argv[2]
    model = load_model(model_path)
    img_tensor = preprocess_image(image_path)
    predict(model, img_tensor)
