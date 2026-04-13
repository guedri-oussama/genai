from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import OPENAI_API_KEY

# Chemins du projet
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = BASE_DIR / "data" / "ai_act.md"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"


def load_document():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_FILE}")

    print(f"Chargement du fichier : {DATA_FILE.name}")

    loader = TextLoader(str(DATA_FILE), encoding="utf-8")
    documents = loader.load()

    # Ajouter un nom de fichier dans les métadonnées
    for doc in documents:
        doc.metadata["filename"] = DATA_FILE.name

    return documents


def main():
    print("=== Étape 1 : Chargement ===")
    documents = load_document()
    print(f"{len(documents)} document(s) chargé(s)\n")

    print("=== Étape 2 : Découpage en chunks ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)
    print(f"{len(split_docs)} chunks créés\n")

    print("=== Étape 3 : Embeddings ===")
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    print("=== Étape 4 : Création FAISS ===")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_STORE_DIR))

    print("\n✅ Base vectorielle créée avec succès !")
    print(f"📁 Dossier : {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()