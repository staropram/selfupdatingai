import json
import torch
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
    # iterate through all messages
    for message in chat_history:
        if message["role"]!="assistant":
            continue


        # look for responses that are just json
        content = message["content"].strip()
        if content.startswith("```json") and content.endswith("```"):
            json_text = content[len("```json"):-3].strip()
            try:
                data = json.loads(json_text)
            except Exception as error:
                print(f"Could not parse training JSON: {error}")
                continue

            # then look for the training signature
            if data.get("type")=="training":
                examples = data.get("examples")
                # and add the examples in
                if examples!=None:
                    training_examples.extend(examples)
    
    return training_examples


def train_model(model, tokenizer, messages, config):
    print("Training model")

    # get the token ids for the messages in the training example
    chat_template = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    input_ids = chat_template["input_ids"]
    
    
    # we need to construct training labels; we don't want to train on our questions
    # only the answers
    assistant_begin_sequence = ["<|im_start|>", "assistant", "Ċ"]
    assistant_begin_ids = tokenizer.convert_tokens_to_ids(assistant_begin_sequence)
    assistant_begin_seqlen = len(assistant_begin_ids)
    answer_starts_index = -1

    # XXX we should generalise this to find all start and end markers for responses
    # find the answer position
    for i in range(0,len(input_ids)-assistant_begin_seqlen+1):
        #input_subseq = input_ids[i:i+assistant_begin_seqlen]
        match_found = True
        for j in range(0,assistant_begin_seqlen):
            if input_ids[i+j]!=assistant_begin_ids[j]:
                match_found = False
                break
        if match_found:
            answer_starts_index = i+assistant_begin_seqlen
            break

    if answer_starts_index == -1:
        raise ValueError("Assistant header not found")

    print(f"answer_starts_index: {answer_starts_index}")

    # make our training labels, we give -100 to the question part
    labels = input_ids.copy()
    labels[:answer_starts_index] = [-100] * answer_starts_index
    print(labels)

    # make the tensors and attach to our GPU, XXX expand this later
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=model.device)
    print(input_tensor.shape)
    print(label_tensor.shape)
    # all tokens are valid
    attention_mask = torch.ones_like(input_tensor)

    # now do the forward pass
    outputs = model(
        input_ids=input_tensor,
        attention_mask=attention_mask,
        labels=label_tensor,
    )

    # check loss
    loss = outputs.loss
    print("Loss:", loss.item())

    # construct our optimiser for BP
    optimiser = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), # generator for mem efficiency
        lr=1.e-4, # learning rate
    )

    # zero the learning gradients
    optimiser.zero_grad()
    # run backprop to compute partial derivs
    loss.backward()
    # apply the weight changes
    optimiser.step()

    model.eval()

    # lets see what happened
    with torch.no_grad():
        outputs = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            labels=label_tensor,
        )

    print("Before:", loss.item())
    print("After:", outputs.loss.item())

    print("end")







