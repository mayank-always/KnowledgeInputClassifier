import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Mapping from category names to integer labels.
label_mapping = {
    "Elementary": 0,
    "Higher Ed": 1,
    "Lifelong Education": 2
}

class TranscriptDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["transcript"]
        label_str = item["category"]
        label = label_mapping[label_str]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Remove the extra batch dimension.
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        return encoding

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # Path to your dataset (ensure dataset.json exists with proper structure)
    DATASET_PATH = "dataset.json"
    # Directory where the trained model and tokenizer will be saved.
    MODEL_SAVE_PATH = "./trained_model"
    
    # Load your transcript dataset.
    data = load_data(DATASET_PATH)
    
    # Use DistilBERT as the base model.
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_mapping)
    )
    
    # Create a PyTorch dataset.
    dataset = TranscriptDataset(data, tokenizer, max_length=512)
    
    # Set training arguments.
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_steps=10,
        evaluation_strategy="no",  # For a small dataset, evaluation is optional.
        logging_dir='./logs',
    )
    
    # Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    # Train the model.
    trainer.train()
    
    # Create the save directory if it doesn't exist.
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    
    # Save the fine-tuned model and tokenizer.
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model and tokenizer have been saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
