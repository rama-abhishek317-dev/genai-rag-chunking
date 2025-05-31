from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOC_PATH = "docs/sample.txt"
PROMPT = "How do you handle employee grievances?"
CHUNK_SETTINGS = [
    {'name': 'Tiny', 'size': 100, 'overlap': 20},
    {'name': 'Ideal', 'size': 300, 'overlap': 50},
    {'name': 'Huge', 'size': 1000, 'overlap': 100},
]

def load_doc(path):
    return TextLoader(path).load()

def run_experiment(name, size, overlap, docs, prompt):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma.from_documents(chunks, embedding)
    retriever = db.as_retriever()
    results = retriever.invoke(prompt)
    print(f'\n=== {name} Chunk ===')
    for i, doc in enumerate(results):
        print(f'[{i+1}] {doc.page_content[:200]}...\n')

documents = load_doc(DOC_PATH)
for config in CHUNK_SETTINGS:
    run_experiment(config['name'], config['size'], config['overlap'], documents, PROMPT)
