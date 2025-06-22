from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
import torch
import base64
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origins in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)
model.load_state_dict(torch.load("best_brain_tumor_resnet18.pth", map_location="cpu"))
model.eval()
target_layers = [model.layer4[-1]]

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        label = "Tumor" if prob > 0.5 else "No Tumor"

    # Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    img_np = np.array(pil_img.resize((224, 224))).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    cam_pil = Image.fromarray(cam_image)
    buffered = io.BytesIO()
    cam_pil.save(buffered, format="PNG")
    cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return JSONResponse(
        {
            "label": label,
            "probability": prob,
            "grad_cam": cam_base64,  # frontend will display this as image/png;base64
        }
    )


app.mount("/static", StaticFiles(directory="frontend/dist/assets"), name="static")


@app.get("/")
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str = ""):
    index_path = os.path.join("frontend", "dist", "index.html")
    return FileResponse(index_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
