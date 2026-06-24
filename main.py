from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("Starting fatigue classification ML pipeline...")

    # 1. Train model
    train_model()

    # 2. Evaluate model
    evaluate_model()

    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
