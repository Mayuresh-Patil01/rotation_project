import torch
import os
from data_loader import get_test_loader
import yaml
from sklearn.metrics import classification_report
from model import get_model

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Check for saved model
if not os.path.exists('best_model.pth'):
    raise FileNotFoundError(
        "Model checkpoint 'best_model.pth' not found. Train first using train.py"
    )

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(config['model_name'], config['num_classes'])
model.load_state_dict(torch.load('best_model.pth'))
model = model.to(device)
model.eval()

# Test loader
test_loader = get_test_loader(config)

all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['0°', '90°', '180°', '270°']))

correct = sum(p == l for p, l in zip(all_preds, all_labels))
total = len(all_labels)
print(f'\nTotal Accuracy: {100 * correct / total:.2f}%')