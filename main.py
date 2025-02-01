from fastapi import FastAPI, File, UploadFile
import torch
from model import get_model
from PIL import Image
import io
import yaml
import uvicorn
from torchvision import transforms

app = FastAPI()

# Load model
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = get_model(config['model_name'], config['num_classes'])
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((config['image_size'], config['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return {"class": predicted.item()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)