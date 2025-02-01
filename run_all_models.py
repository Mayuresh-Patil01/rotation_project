import yaml
import subprocess
import argparse
from pathlib import Path

MODELS = ["basic", "vgg", "resnet", "alexnet", "inception", "vit"]
CONFIG_FILE = "config.yaml"
LOG_DIR = "logs"

def update_config(model_name: str):
    """Update config.yaml with current model settings"""
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    config['model_name'] = model_name
    config['image_size'] = 299 if model_name == "inception" else 224
    
    with open(CONFIG_FILE, 'w') as f:
        yaml.dump(config, f)

def run_experiment(model_name: str):
    """Run training and testing for a single model"""
    print(f"\n{'='*40}")
    print(f"Starting experiment for {model_name.upper()}")
    print(f"{'='*40}")
    
    # Create log directory
    log_dir = Path(LOG_DIR) / model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training with logging
    with open(log_dir / "train.log", 'w') as f:
        train_process = subprocess.run(
            ["python", "train.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    # Run testing with logging
    with open(log_dir / "test.log", 'w') as f:
        test_process = subprocess.run(
            ["python", "test.py"],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    return {
        'model': model_name,
        'train_exit': train_process.returncode,
        'test_exit': test_process.returncode
    }

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=MODELS,
                        help="List of models to test")
    args = parser.parse_args()

    # Create log directory
    Path(LOG_DIR).mkdir(exist_ok=True)
    
    results = []
    for model in args.models:
        try:
            # Update configuration
            update_config(model)
            
            # Run experiment
            result = run_experiment(model)
            results.append(result)
            
            # Print summary
            print(f"\n{model.upper()} Results:")
            print(f"Training exit code: {result['train_exit']}")
            print(f"Testing exit code: {result['test_exit']}")
            
        except Exception as e:
            print(f"Error running {model}: {str(e)}")
            results.append({'model': model, 'error': str(e)})
    
    # Save final report
    with open(Path(LOG_DIR) / "summary.txt", 'w') as f:
        f.write("Experiment Summary:\n")
        f.write("="*40 + "\n")
        for result in results:
            f.write(f"{result['model']}:\n")
            f.write(f"  Training exit code: {result.get('train_exit', 'N/A')}\n")
            f.write(f"  Testing exit code: {result.get('test_exit', 'N/A')}\n")
            if 'error' in result:
                f.write(f"  ERROR: {result['error']}\n")
            f.write("\n")

if __name__ == "__main__":
    main()