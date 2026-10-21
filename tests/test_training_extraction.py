from selfupdatingai.training import extract_training_examples_from_file
import json

def test_training_extraction():
    file_name = "tests/test_data/test_training_example_01.json"
    training_examples = extract_training_examples_from_file(file_name)
    print(training_examples)


if __name__ == "__main__":
    test_training_extraction()