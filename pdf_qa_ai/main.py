from extract_text import extract_text_from_pdf
from chunker import chunk_text
from embedder import embed_chunks, embed_question
from retriever import build_index, find_top_chunks
from answer_generator import generate_answer

def main():
    file_path = "sample-policy.pdf"  # Replace with your actual file if different
    text = extract_text_from_pdf(file_path)

    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)
    index = build_index(chunk_embeddings)

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()
        if question.lower() == "exit":
            break

        question_embedding = embed_question(question)
        top_chunks = find_top_chunks(question_embedding, chunk_embeddings, chunks)

        # Generate a real answer using a local LLM
        combined_context = "\n".join(top_chunks)
        answer = generate_answer(question, combined_context)

        print(f"\n🤖 Answer:\n{answer}\n")

if __name__ == "__main__":
    main()
# pdf_qa_ai/main.py
# This script orchestrates the PDF QA AI pipeline, extracting text, chunking it,                                                    