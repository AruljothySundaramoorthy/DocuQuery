from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Use a small, fast model for question-answering
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def generate_answer(question, context):
    prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    result = qa_pipeline(prompt, max_new_tokens=100)[0]["generated_text"]
    return result.strip()
