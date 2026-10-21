import json
from pathlib import Path

def extract_training_examples_from_file(file_name):
    try:
        with open(file_name,"r",encoding="utf-8") as f:
            chat_history = json.load(f)
    except Exception:
        chat_history = json.loads("[{\"role\":\"user\",\"content\":\"hi\"}]")

    return extract_training_examples(chat_history)
    

def extract_training_examples(chat_history):
    print("extracting training examples")
    training_examples = []
    for message in chat_history:
        if message["role"]!="assistant":
            continue


        content = message["content"].strip()
        if content.startswith("```json") and content.endswith("```"):
            json_text = content[len("```json"):-3].strip()
            try:
                data = json.loads(json_text)
                training_examples.append(data)
            except Exception as error:
                print(error)
    
    return training_examples