# GenAI RAG Chunking Experiment

This project explores how different chunk sizes affect document retrieval in Retrieval-Augmented Generation (RAG) systems.

## 📦 Installation

```bash
python -m venv genrag
source genrag/bin/activate  # or .\genrag\Scripts\activate on Windows
pip install -r requirements.txt
```

## ⚙️ Setup & Usage

1. Add your documents to the `docs/` folder.
2. Run `chunk_experiment.py` to compare chunk strategies.

```bash
python chunk_experiment.py
```

Or use the Jupyter notebook version:

```bash
jupyter notebook chunk_experiment.ipynb
```

## 🔍 Chunk Experiment Instructions

The scripts run retrieval on a prompt using three configurations:

- Tiny Chunk (100, 20)
- Ideal Chunk (300, 50)
- Huge Chunk (1000, 100)

Prompt used:
```
How do you handle employee grievances?
```

## ✍️ Credits & License

Created by Abhishek Rama with ❤️ using LangChain + ChromaDB.
MIT License.
