import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel,LoraConfig,get_peft_model
from datetime import datetime
import json
import random
import string

import selfupdatingai.training as training

# This handles interfacing with the local chat model
class AIInterface:

    # the ai interface keeps track of the current conversation and saves it when requested
    def save_current_chat(self):

        # deal with chat naming when there isn't any
        if self.current_chat_filename is None:
            chat_id = "".join(random.choices(string.ascii_letters + string.digits, k=5)) 
            today_string = datetime.now().strftime("%d_%m_%Y")
            self.current_chat_filename = f"chats/{chat_id}_{today_string}.json"

        print(f"saving {self.current_chat_filename}")

        # save as json
        with open(self.current_chat_filename, "w", encoding="utf-8") as f:
            json.dump(self.message_list, f, indent=2, ensure_ascii=False)

    # this loads the chat pointed to by chat_filename and makes it the current chat
    def load_current_chat(self,chat_filename):
        try:
            with open(chat_filename,"r",encoding="utf-8") as f:
                self.message_list = json.load(f)
            # if that worked make sure we set the current chat filename
            self.current_chat_filename = chat_filename
        except Exception as error:
            self.message_list = [
                {"role": "assistant", "content": f"Could not load chat: {error}"}
            ]

        # dont return the system msg
        if self.message_list and self.message_list[0]["role"] == "system":
            return self.message_list[1:]

        return self.message_list


    def load_lora_adapter(self):
        print(f"Loading existing LoRA adapter: {self.adapter_name}")
        return PeftModel.from_pretrained(
            self.base_model,
            f"adapters/{self.adapter_name}",
            is_trainable=True
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

    # this sends a message to the ai model, i.e runs a forward pass
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
            max_new_tokens=self.max_new_tokens,
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

    def reset_message_list(self):
        self.message_list = []

    def __init__(self,config):
        print("Initialising AIInterface")

        # extract soem params from the config
        self.model_name = config["model"]["name"]
        self.adapter_name = config["model"]["adapter"]
        self.max_new_tokens = config["generation"]["max_new_tokens"]

        # setup the initial system prompt (XXX load this later)
        #self.message_list = [{"role":"system","content":"You are a self-learning AI"}]
        self.message_list = []

        # start with an empty chat filename
        self.current_chat_filename = None

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

         #this is completely pointless at this point
        if self.adapter_name=="":
            print("No adapter supplied, using base model instead")
            self.model = self.base_model
        else:
            adapter_path = Path(f"adapters/{self.adapter_name}")
            if adapter_path.exists():
                self.model = self.load_lora_adapter()
            else:
                self.model = self.create_lora_adapter()

    def train_model(self,examples,training_config):
        training.train_model(
            self.model,
            self.tokenizer,
            examples,
            training_config
        ) 

    # saves the peft model, the LORA adapter
    def save_adapter(self,adapter_name):
        self.model.save_pretrained(f"adapters/{adapter_name}")
