import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config
from api import run_app


def main():
    parser = argparse.ArgumentParser(description="AI Inference Service")
    parser.add_argument(
        "command",
        choices=["serve", "train"],
        help="Command to run",
    )
    parser.add_argument("--host", default=None, help="API host")
    parser.add_argument("--port", type=int, default=None, help="API port")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--use-cifar10", action="store_true", help="Use CIFAR-10 dataset for training")
    
    args = parser.parse_args()
    
    Config.ensure_dirs()
    
    if args.command == "serve":
        run_app(host=args.host, port=args.port, debug=args.debug)
    
    elif args.command == "train":
        from classification import ClassifierTrainer
        from data import create_data_loaders
        from utils import plot_training_curves
        
        num_epochs = args.epochs or Config.NUM_EPOCHS
        batch_size = args.batch_size or Config.BATCH_SIZE
        lr = args.lr or Config.LEARNING_RATE
        
        print("Creating data loaders...")
        loaders = create_data_loaders(
            batch_size=batch_size,
            num_workers=Config.NUM_WORKERS,
            image_size=Config.IMAGE_SIZE,
            use_cifar10=args.use_cifar10,
        )
        
        if "train" not in loaders or "val" not in loaders:
            print("Error: Training or validation data not available.")
            print("Use --use-cifar10 to train on CIFAR-10 dataset,")
            print("or provide train/val directories.")
            return
        
        print(f"Training for {num_epochs} epochs...")
        trainer = ClassifierTrainer(
            num_classes=len(Config.CLASS_NAMES),
            learning_rate=lr,
            device=Config.DEVICE,
            save_path=Config.CLASSIFICATION_MODEL_PATH,
            pretrained=True,
        )
        
        history = trainer.train(loaders["train"], loaders["val"], num_epochs=num_epochs)
        
        curves_path = os.path.join(Config.OUTPUTS_DIR, "training_curves.png")
        plot_training_curves(history, save_path=curves_path)
        print(f"Training curves saved to {curves_path}")
        
        print("Training complete!")


if __name__ == "__main__":
    main()
