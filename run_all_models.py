import yaml
import subprocess
import argparse
from pathlib import Path
import os
import sys
from pathlib import Path

MODELS = ["basic", "vgg", "resnet", "alexnet", "inception", "vit"]  
CONFIG_FILE = "config.yaml"
LOG_DIR = "logs"

def update_config(model_name: str):
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    config['model_name'] = model_name
    config['image_size'] = 299 if model_name == "inception" else 224
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)

def run_experiment(model_name: str):
    print(f"\n{'='*40}")
    print(f"Training {model_name.upper()} (Image Size: {299 if model_name == 'inception' else 224})")
    print(f"{'='*40}")
    
    log_dir = Path(LOG_DIR) / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    update_config(model_name)
    
    # Training with error checking
    train_success = False
    with open(log_dir / "train.log", 'w') as f:
        result = subprocess.run(
            ["python", "train.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
        train_success = (result.returncode == 0)
    
    # Testing only if training succeeded
    if train_success and Path('best_model.pth').exists():
        with open(log_dir / "test.log", 'w') as f:
            subprocess.run(
                ["python", "test.py"],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        Path('best_model.pth').unlink()  # Cleanup
        return True
    else:
        print(f"❌ {model_name.upper()} failed during training")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=MODELS,
                       help="Models to test (default: all)")
    args = parser.parse_args()

    Path(LOG_DIR).mkdir(exist_ok=True)
    
    for model in args.models:
        try:
            run_experiment(model)
            print(f"✅ {model.upper()} completed")
        except Exception as e:
            print(f"❌ {model.upper()} failed: {str(e)}")
    
    print("\nExperiment summary saved in logs/ directory")

if __name__ == "__main__":
    main()