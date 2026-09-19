import sys
import os

# Allow running this script directly (adds project root to sys.path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import load_and_chunk
from src.vectorstore import build_vectorstore


def main():
    print("Loading and chunking claim PDFs...")
    documents = load_and_chunk()

    print("\nBuilding ChromaDB vector store (this may take a minute on first run)...")
    build_vectorstore(documents, reset=True)

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
