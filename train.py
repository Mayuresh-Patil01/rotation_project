import yaml
import torch
from data_loader import get_loaders
from model import get_model

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader = get_loaders(config)
    model = get_model(config['model_name'], config['num_classes'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f'Validation Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()