import gradio as gr
from pathlib import Path
import selfupdatingai.training as training


class HTMLUI:
    def __init__(self,ai_interface=None):
        self.ai_interface = ai_interface
        self.chat_names = self.get_chat_names()
        with gr.Blocks(title="Self-Updating AI") as self.app:
            self.build_interface()
            self.connect_events()

    def get_chat_names(self):
        files = list(Path("chats").glob("*.json"))
        return [f.stem for f in files]

    
    def chat_accept_message(self, message, history):
        if not message.strip():
            return history, message, "Nothing to send."

        history = history or []
        history.append({
            "role": "user",
            "content": message,
        })
        response = self.ai_interface.send_message(message)
        history.append({
            "role":"assistant",
            "content":response
        })

        self.ai_interface.save_current_chat()

        return history, "", "Model responded."

    def build_interface(self):
        gr.Markdown("# Self-Updating AI")
        gr.Markdown("Model: Qwen3-4B-Instruct-2507")

        # keep track of available chats
        self.available_chat_buttons = []

        # sub method the tabs
        self.build_chat_tab()
        self.build_training_tab()
        self.build_inspection_tab()

    def connect_events(self):
        self.connect_chat_events()
        self.connect_training_events()


    def build_chat_tab(self):
        with gr.Tab("Chat"):
            # make buttons to select historical chats
            with gr.Sidebar() as self.chat_selection_sidebar:
                for chat_name in self.chat_names:
                    button = gr.Button(f"{chat_name}",variant="secondary",size="sm")
                    self.available_chat_buttons.append(button)

            self.chatbot = gr.Chatbot(
                label="Conversation",
                height=600,
            )

            with gr.Row():
                self.chat_message_box = gr.Textbox(
                    placeholder="Write a message...",
                    label="Message",
                    lines=2,
                    scale=8,
                    interactive=True
                )

                self.chat_send_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                )

            with gr.Column():
                self.chat_clear_button = gr.ClearButton(
                    [self.chat_message_box, self.chatbot],
                    value="Clear",
                )

            self.chat_status_box = gr.Textbox(
                value="Ready.",
                label="Status",
                interactive=False,
            )



    def build_inspection_tab(self):
        with gr.Tab("Inspect"):
            gr.Markdown("Inspection tab")
    

    def connect_chat_events(self):

        print("Wiring buttons:", len(self.available_chat_buttons))


        for button in self.available_chat_buttons:
            button.click(
                fn=self.load_selected_chat,
                inputs=[button],
                outputs=[self.chatbot, self.chat_status_box],
            )

        self.chat_send_button.click(
            fn=self.chat_accept_message,
            inputs=[self.chat_message_box, self.chatbot],
            outputs=[self.chatbot, self.chat_message_box, self.chat_status_box],
        )

        self.chat_message_box.submit(
            fn=self.chat_accept_message,
            inputs=[self.chat_message_box, self.chatbot],
            outputs=[self.chatbot, self.chat_message_box, self.chat_status_box],
        )

        self.chat_clear_button.click(
            fn=lambda: "Ready.",
            outputs=self.chat_status_box,
        )

    def build_training_tab(self):
        self.training_chat_selector_buttons = []

        with gr.Tab("Train"):
            gr.Markdown("Training tab")


            # make buttons to select historical chats
            with gr.Sidebar() as self.chat_selection_sidebar:
                for chat_name in self.chat_names:
                    button = gr.Button(f"{chat_name}",variant="secondary",size="sm")
                    self.training_chat_selector_buttons.append(button)

            self.training_checkboxes = []

            self.training_data_selector = gr.Textbox(
                value="Ready.",
                label="Placeholder for data selector",
                interactive=False,
            )

            #for i, example in enumerate(training_examples):
            #    with gr.Column():
            #        checkbox = gr.Checkbox(label=f"Example {i + 1}")
            #        self.training_checkboxes.append(checkbox)
#
#                    with gr.Accordion("View JSON", open=False):
#                        gr.JSON(value=example)

    def connect_training_events(self):
        for button in self.training_chat_selector_buttons:
            button.click(
                fn=self.training_extract_data,
                inputs=[button],
                outputs=[self.training_data_selector],
            )

        print("connecting training events")

    def training_extract_data(self,chat_file_stem):
        file_name = f"chats/{chat_file_stem}.json"
        training_examples = training.extract_training_examples_from_file(file_name)
        print("extracting training data")
        print(training_examples)
        return training_examples

    def load_selected_chat(self,chat_name):
        chat_filename = f"chats/{chat_name}.json"
        history = self.ai_interface.load_current_chat(chat_filename)

        return [history,f"Loaded chat {chat_name}"]


    def launch(self):
        self.app.launch(theme='Taithrah/Minimal')




if __name__ == "__main__":
    HTMLUI().launch()