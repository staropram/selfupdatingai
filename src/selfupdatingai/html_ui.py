import gradio as gr
from pathlib import Path


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

    
    def accept_message(self, message, history):
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

        with gr.Tab("Chat"):
            with gr.Sidebar() as self.chat_selection_sidebar:

                # make buttons to select historical chats
                for chat_name in self.chat_names:
                    button = gr.Button(f"{chat_name}",variant="secondary",size="sm")
                    self.available_chat_buttons.append(button)

            self.chatbot = gr.Chatbot(
                label="Conversation",
                height=600,
            )

            with gr.Row():
                self.message_box = gr.Textbox(
                    placeholder="Write a message...",
                    label="Message",
                    lines=2,
                    scale=8,
                    interactive=True
                )

                self.send_button = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                )

            with gr.Column():
                self.clear_button = gr.ClearButton(
                    [self.message_box, self.chatbot],
                    value="Clear",
                )

            self.status_box = gr.Textbox(
                value="Ready.",
                label="Status",
                interactive=False,
            )

        with gr.Tab("Train"):
            gr.Markdown("Training tab")

        with gr.Tab("Inspect"):
            gr.Markdown("Inspection tab")

    def connect_events(self):

        print("Wiring buttons:", len(self.available_chat_buttons))


        for button in self.available_chat_buttons:
            button.click(
                fn=self.load_selected_chat,
                inputs=[button],
                outputs=[self.chatbot, self.status_box],
            )

        self.send_button.click(
            fn=self.accept_message,
            inputs=[self.message_box, self.chatbot],
            outputs=[self.chatbot, self.message_box, self.status_box],
        )

        self.message_box.submit(
            fn=self.accept_message,
            inputs=[self.message_box, self.chatbot],
            outputs=[self.chatbot, self.message_box, self.status_box],
        )

        self.clear_button.click(
            fn=lambda: "Ready.",
            outputs=self.status_box,
        )

    def load_selected_chat(self,chat_name):
        chat_filename = f"chats/{chat_name}.json"
        history = self.ai_interface.load_current_chat(chat_filename)

        return [history,f"Loaded chat {chat_name}"]


    def launch(self):
        self.app.launch(theme='Taithrah/Minimal')

if __name__ == "__main__":
    HTMLUI().launch()