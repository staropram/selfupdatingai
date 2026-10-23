from selfupdatingai.ai_interface import AIInterface
from selfupdatingai.training import extract_training_examples_from_file
import json
from pathlib import Path
import tomllib

def test_model_training():
    print("Testing model training")
    # load a training example
    file_name = "tests/test_data/test_training_example_02.json"
    training_examples = extract_training_examples_from_file(file_name)
    first_training_example = training_examples[0]["messages"]

    # load config
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    CONFIG_PATH = PROJECT_ROOT / "config.toml"
    with CONFIG_PATH.open("rb") as config_file:
        config = tomllib.load(config_file)

    # use the blank adapter to start with
    config["model"]["adapter"] = "blank_adapter"

    # load ai interface
    ai_interface = AIInterface(config)

    # see what it says prior to fine tuning
    name_question = "In one word, what is your name?"
    name_response_1 = ai_interface.send_message(name_question,deterministic=True)
    print(f"Initial response to name question: {name_response_1}")


    # manual validation
    token_ids = ai_interface.tokenizer.apply_chat_template(
        first_training_example,
        tokenize=True,
        add_generation_prompt=False,
    )
    print(ai_interface.tokenizer.convert_ids_to_tokens(token_ids["input_ids"]))

    # now actually train the model
    ai_interface.train_model(first_training_example,None)

    # see what it says prior to fine tuning
    ai_interface.reset_message_list()
    name_response_2 = ai_interface.send_message(name_question,deterministic=True)
    print(f"Final response to name question: {name_response_2}")

    print("Saving model")
    ai_interface.save_adapter("bongoV1")


if __name__ == "__main__":
    test_model_training()