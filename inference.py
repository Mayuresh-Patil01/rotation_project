import torch
from model import get_model
from PIL import Image
import matplotlib.pyplot as plt

def predict(image_path, model_path='best_model.pth'):
    model = get_model('basic', num_classes=4)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    class_names = ['0°', '90°', '180°', '270°']
    pred_class = class_names[torch.argmax(output).item()]
    
    plt.imshow(image)
    plt.title(f'Predicted: {pred_class}')
    plt.show()

if __name__ == '__main__':
    predict('test_image.jpg')