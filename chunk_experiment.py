from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOC_PATH = "docs/sample.txt"
PROMPT = "How do you handle employee grievances?"
IDEAL_RESPONSE = (
    "If an employee faces any issue or grievance, they should report it confidentially to the HR team. "
    "The HR team will acknowledge within 24 hours and initiate an internal resolution process. "
    "Most issues are resolved within 5 working days."
)

CHUNK_SETTINGS = [
    {'name': 'Tiny', 'size': 100, 'overlap': 20},
    {'name': 'Ideal', 'size': 300, 'overlap': 50},
    {'name': 'Huge', 'size': 1000, 'overlap': 100},
]

def load_doc(path):
    return TextLoader(path, encoding='utf-8').load()

def run_experiment(name, size, overlap, docs, prompt):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    
    print(f"\n{'='*20} {name} Chunking (Chunk Size={size}, Overlap={overlap}) {'='*20}")
    print(f"🔹 Total Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:\n{chunk.page_content[:500]}...\n")

    embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma.from_documents(chunks, embedding)
    retriever = db.as_retriever()
    results = retriever.invoke(prompt)
    actual_response = "\n".join([doc.page_content.strip() for doc in results])

    print(f"\n{'-' * 50}")
    print(f"{name} Chunk Size")
    print(f"Prompt: {prompt}\n")
    print(f"Ideal Response:\n{IDEAL_RESPONSE}\n")
    print(f"Actual Response due to {name.lower()} chunking:\n{actual_response[:1000]}")  # Trim for clarity

documents = load_doc(DOC_PATH)
for config in CHUNK_SETTINGS:
    run_experiment(config['name'], config['size'], config['overlap'], documents, PROMPT)
