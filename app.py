import gradio as gr
from rag_pipeline import load_contract, query_contract

#handling file uploads
def upload_file(file):
    if file is None:
        return "⚠️ Please upload a file."
    load_contract(file)
    return "✅ Contract uploaded and indexed successfully!"

#chatbot function
def chat_with_bot(message, history):
    response = query_contract(message)
    # Adapt response into Gradio-friendly format
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]
    return history


with gr.Blocks(title="Legal Contract Assistant") as demo:
    gr.Markdown("## 📜 Legal Contract Summarizer + Explainer + Q&A (RAG powered)")

    with gr.Row():
        file_upload = gr.File(label="Upload Contract", type="filepath")
        upload_status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Chat with your contract", type="messages")
    user_input = gr.Textbox(label="Your question")

    with gr.Row():
        send_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear Chat")

    file_upload.upload(fn=upload_file, inputs=file_upload, outputs=upload_status)
    send_btn.click(fn=chat_with_bot, inputs=[user_input, chatbot], outputs=chatbot)
    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
