from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docs = {
    "Delhi is the capital of India.",
    "Mumbai is the capital of Maharashtra.",
    "Kolkata is the capital of West Bengal.",
    "Chennai is the capital of Tamil Nadu."
}
vector = embedding.embed_documents(docs)

print(str(vector))