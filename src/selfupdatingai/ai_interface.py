import torch
import tomllib
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel,LoraConfig,get_peft_model

# the basic Ai interface
class AIInterface:

    def load_lora_adapter(self):
        print(f"Loading existing LoRA adapter: {self.adapter_name}")
        return PeftModel.from_pretrained(
            self.base_model,
            f"adapters/{self.adapter_name}",
        )

    def create_lora_adapter(self):
        print("Initialising empty LoRA adapter")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        peft_model = get_peft_model(
            self.base_model,
            lora_config,
        )

        # save weights on init
        peft_model.save_pretrained(f"adapters/{self.adapter_name}")

        return peft_model

    def send_message(self,new_message,deterministic=False):
        user_message = {
            "role": "user",
            "content": new_message
        }
        self.message_list.append(user_message)

        prompt = self.tokenizer.apply_chat_template(
            self.message_list,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=deterministic
        )

        input_length = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, input_length:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        self.message_list.append({
            "role":"assistant",
            "content":response
        })

        return response

    def __init__(self):
        print("Initialising AIInterface")
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        CONFIG_PATH = PROJECT_ROOT / "config.toml"

        with CONFIG_PATH.open("rb") as config_file:
            config = tomllib.load(config_file)

        self.model_name = config["model"]["name"]

        self.adapter_name = config["model"]["adapter"]

        self.message_list = []

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )

        if self.adapter_name=="":
            print("No adapter supplied, using base model instead")
            self.model = self.base_model
        else:
            adapter_path = Path(f"adapters/{self.adapter_name}")
            if adapter_path.exists():
                self.model = self.load_lora_adapter()
            else:
                self.model = self.create_lora_adapter()
