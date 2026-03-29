"""
ASG 04 – Main Pipeline
Orchestrates the entire machine learning workflow.
"""

from src.data_ingestion import ingest_data
from src.pre_processing import preprocess
from src.train import train
from src.evaluation import evaluate

def run_main_pipeline():
    print("Memulai Pipeline Spaceship Titanic")
    print("_" * 30)

    ingest_data()

    train_data, test_data, preprocessor = preprocess()

    pipeline = train(train_data, preprocessor)
    accuracy, precision, recall = evaluate(test_data, pipeline)
    
    print("berhasil")
    print(f"accuracy: {accuracy:.4f}")
    
    if accuracy > 0.70:
        print("Status: layak.")
    else:
        print("Status: belum mumpuni.")

if __name__ == "__main__":
    run_main_pipeline()