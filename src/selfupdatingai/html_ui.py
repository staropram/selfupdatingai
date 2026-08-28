import gradio as gr


class HTMLUI:
    def __init__(self,ai_interface):
        self.ai_interface = ai_interface
        with gr.Blocks(title="Self-Updating AI") as self.app:
            self.build_interface()
            self.connect_events()

    
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

        return history, "", "Model responded."

    def build_interface(self):
        gr.Markdown("# Self-Updating AI")
        gr.Markdown("Model: Qwen3-4B-Instruct-2507")

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

        self.clear_button = gr.ClearButton(
            [self.message_box, self.chatbot],
            value="Clear conversation",
        )

        self.status_box = gr.Textbox(
            value="Ready.",
            label="Status",
            interactive=False,
        )

    def connect_events(self):
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

    def launch(self):
        self.app.launch()

if __name__ == "__main__":
    HTMLUI().launch()