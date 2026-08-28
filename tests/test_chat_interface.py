import gradio as gr

def accept_message(message, history):
    if not message.strip():
        return history, message, "Nothing to send."

    history = history or []
    history.append({
        "role": "user",
        "content": message,
    })

    return history, "", "Message received. Model connection not added yet."

with gr.Blocks(title="Self-Updating AI") as app:
    gr.Markdown("# Self-Updating AI")
    gr.Markdown("Model: Qwen3-4B-Instruct-2507")

    chatbot = gr.Chatbot(
        label="Conversation",
        height=600,
    )

    with gr.Row():
        message_box = gr.Textbox(
            placeholder="Write a message...",
            label="Message",
            lines=2,
            scale=8,
            interactive=True
        )

        send_button = gr.Button(
            "Send",
            variant="primary",
            scale=1,
        )

    status_box = gr.Textbox(
        value="Ready.",
        label="Status",
        interactive=False,
    )

    send_button.click(
        fn=accept_message,
        inputs=[message_box, chatbot],
        outputs=[chatbot, message_box, status_box],
    )

    message_box.submit(
        fn=accept_message,
        inputs=[message_box, chatbot],
        outputs=[chatbot, message_box, status_box],
    )

    clear_button = gr.ClearButton(
        [message_box, chatbot],
        value="Clear conversation",
    )

    clear_button.click(
        fn=lambda: "Ready.",
        outputs=status_box,
    )



if __name__ == "__main__":
    app.launch()