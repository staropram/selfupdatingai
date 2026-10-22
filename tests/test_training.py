from selfupdatingai.ai_interface import AIInterface
from selfupdatingai.training import extract_training_examples_from_file
import json

def test_model_training():
    print("Testing model training")
    # load a training example
    file_name = "tests/test_data/test_training_example_02.json"
    training_examples = extract_training_examples_from_file(file_name)
    first_training_example = training_examples[0]["messages"]

    # load ai interface
    ai_interface = AIInterface()

    # manual validation
    token_ids = ai_interface.tokenizer.apply_chat_template(
        first_training_example,
        tokenize=True,
        add_generation_prompt=False,
    )
    print(ai_interface.tokenizer.convert_ids_to_tokens(token_ids["input_ids"]))

    # now actually train the model
    ai_interface.train_model(first_training_example,None)


if __name__ == "__main__":
    test_model_training()