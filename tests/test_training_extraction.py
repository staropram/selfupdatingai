from selfupdatingai.training import extract_training_examples
import json

def test_training_extraction():
    with open("tests/test_data/test_training_example_01.json","r",encoding="utf-8") as f:
        training_example = json.load(f)
    training_examples = extract_training_examples(training_example)
    print("ok")


if __name__ == "__main__":
    test_training_extraction()