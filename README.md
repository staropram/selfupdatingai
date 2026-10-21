In this project we are experimenting with AI that can modify themselves using LoRA adapters, and we want to visualise how interior aspects of the model change when this happens. The idea is we see what the model's "intrinsic" trajectory does to itself. Does this only amplify apriori "basined" behaviour or do we get odd divergence? Then we can deliberatey introduce divergence. Then we want to see how the model internals differ in the inspection tab.

I'm not using AI to code this because that's boring; thinking from the inside of something is far more useful even if it is slow.

## Chat interface 

The basics work, you can chat to a model, it automatically saves conversations.

TODO

- Ability to select the base model and LORA adapter from the interface instead of relying on a config file
- Edit loaded chats
- Rename chats

## Training interface

Nothing significant is done yet except the parser can extract training examples from chats generated in the other tab.

TODO

- A sensible layout to select which training examples to use, set training options etc
- The training mechanism itself with suitable UI feedback


## Inspection interface

My current thinking is that this will allow you to select the model and adapter and present queries. We will do things like plot the trajectory of the query through state space using a collapsed representation, and then compare that trajectory before and after LORA adaptation. I don't actually know yet and that's part of the fun, I'm exploring here.
