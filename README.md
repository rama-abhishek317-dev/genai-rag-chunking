# GenAI RAG Chunking Experiment

This project demonstrates how chunk size affects information retrieval quality in a RAG (Retrieval-Augmented Generation) pipeline using LangChain and HuggingFace.

## 📚 What is RAG?

RAG enhances LLMs by retrieving relevant documents from a vector store before generating answers. It's especially useful when working with private or domain-specific knowledge.

## 🎯 Project Use Case

We simulate an enterprise HR document and test how different chunking strategies (Tiny, Ideal, Huge) affect response accuracy when querying: _"How do you handle employee grievances?"_

## 🧪 Chunking Comparison Experiment

We test 3 configurations:

| Name  | Chunk Size | Overlap |
|-------|------------|---------|
| Tiny  | 100        | 20      |
| Ideal | 300        | 50      |
| Huge  | 1000       | 100     |

For each configuration, we:

- Show total number of chunks and their contents
- Retrieve response for the prompt
- Compare it against the ideal answer

### ✅ Ideal Response

```
If an employee faces any issue or grievance, they should report it confidentially to the HR team. 
The HR team will acknowledge within 24 hours and initiate an internal resolution process. 
Most issues are resolved within 5 working days.
```

## 📂 Folder Structure

```
genai_rag/
├── docs/
│   └── sample.txt
├── chunk_experiment.py
├── chunk_experiment.ipynb
├── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## 📦 Installation

```bash
git clone https://github.com/<your-username>/genai-rag-chunking.git
cd genai_rag_chunking
python -m venv genrag
genrag\Scripts\activate   # on Windows
pip install -r requirements.txt
```

## ⚙️ Setup & Usage

1. Place your documents in `docs/` folder.
2. Edit the prompt in `chunk_experiment.py` if needed.
3. Run the script:

```bash
python chunk_experiment.py
```

Or launch Jupyter Notebook:

```bash
jupyter notebook chunk_experiment.ipynb
```

## ✍️ Credits & License

- Built with LangChain, HuggingFace Transformers, and ChromaDB
- MIT License