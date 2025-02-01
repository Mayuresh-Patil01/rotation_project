import torch
from model import get_model
from data_loader import get_loaders
import yaml
from sklearn.metrics import classification_report  # Now installed

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

_, test_loader = get_loaders(config)
model = get_model(config['model_name'], config['num_classes'])
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['0°', '90°', '180°', '270°']))

correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l)
total = len(all_labels)
print(f'\nTotal Accuracy: {100 * correct / total:.2f}%')