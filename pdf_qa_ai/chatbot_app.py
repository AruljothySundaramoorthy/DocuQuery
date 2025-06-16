import gradio as gr
from extract_text import extract_text_from_pdf
from chunker import chunk_text
from embedder import embed_chunks, embed_question
from retriever import build_index, find_top_chunks
from answer_generator import generate_answer

stored_chunks = []
chunk_embeddings = []
file_loaded = False

def load_pdf(file):
    global stored_chunks, chunk_embeddings, file_loaded
    if file is None:
        return "❌ Please upload a PDF first."
    text = extract_text_from_pdf(file.name)
    stored_chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(stored_chunks)
    file_loaded = True
    return f"✅ File '{file.name}' processed. You can now ask questions."

def chat_interface(user_input, history):
    if not file_loaded:
        return history + [[user_input, "❌ Please upload a file first."]]
    question_embedding = embed_question(user_input)
    top_chunks = find_top_chunks(question_embedding, chunk_embeddings, stored_chunks)
    context = "\n".join(top_chunks)
    answer = generate_answer(user_input, context)
    history.append([user_input, answer])
    return history

with gr.Blocks() as demo:
    gr.Markdown("# 📄 AskMyPDF: Chat with your PDF")
    file_input = gr.File(label="Upload a PDF", file_types=[".pdf"])
    load_button = gr.Button("Load PDF")
    status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your question")
    send_btn = gr.Button("Send")

    load_button.click(load_pdf, inputs=file_input, outputs=status)
    send_btn.click(chat_interface, inputs=[msg, chatbot], outputs=chatbot)
    msg.submit(chat_interface, inputs=[msg, chatbot], outputs=chatbot)

if __name__ == "__main__":
    demo.launch()
