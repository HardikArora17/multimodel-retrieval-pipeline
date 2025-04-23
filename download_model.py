from transformers import AutoModel, AutoTokenizer
import os

def download_and_load_model(model_name: str, save_directory: str = "downloaded_models"):
    os.makedirs(save_directory, exist_ok=True)

    # Download and save
    print(f"Downloading model and tokenizer for '{model_name}'...")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to '{save_directory}'")

    # Load back from local
    print(f"Loading model and tokenizer from '{save_directory}'...")
    local_model = AutoModel.from_pretrained(save_directory)
    local_tokenizer = AutoTokenizer.from_pretrained(save_directory)
    print("LOADED SUCCESSFULLY")

    print(local_model)

#model_name = 'facebook/bart-large-cnn'
model_name = 'meta-llama/Llama-3.1-8B-Instruct'

download_and_load_model(model_name = model_name, save_directory = "downloaded_models")
