from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from app.config import OPENAI_API_KEY

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_DIR = BASE_DIR / "vector_store"


def format_context(docs):
    parts = []

    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("filename", "fichier inconnu")
        source = doc.metadata.get("source", "source inconnue")
        content = doc.page_content.strip()

        parts.append(
            f"[Source {i} | fichier: {filename} | source: {source}]\n{content}"
        )

    return "\n\n".join(parts)


def main():
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            "La base vectorielle n'existe pas encore. Lance d'abord build_vectorstore."
        )

    print("Chargement de la base vectorielle...")

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template(
        """Tu es un assistant spécialisé sur le règlement européen sur l'intelligence artificielle (AI Act).

Règles STRICTES :
- Réponds toujours en français.
- Réponds uniquement à partir du contexte fourni.
- N'invente rien.
- Si l'information n'est pas dans le contexte, dis clairement :
  "Je ne trouve pas l'information dans le document."
- Termine toujours par une section :

Sources :
- [Source 1]
- [Source 2]

Contexte :
{context}

Question :
{question}

Réponse :
"""
    )

    chain = prompt | llm

    print("RAG chatbot AI Act prêt. Tape 'quit' pour sortir.\n")

    while True:
        question = input("Toi : ")

        if question.lower() in ["quit", "exit", "q"]:
            print("Fin du chatbot.")
            break

        docs = retriever.invoke(question)
        context = format_context(docs)

        response = chain.invoke({
            "context": context,
            "question": question
        })

        print("\n--- Sources retrouvées ---")
        for i, doc in enumerate(docs, start=1):
            filename = doc.metadata.get("filename", "fichier inconnu")
            source = doc.metadata.get("source", "source inconnue")
            print(f"{i}. {filename} | {source}")

        print(f"\nAssistant :\n{response.content}\n")


if __name__ == "__main__":
    main()