# ⚖️ Chatbot Expert AI Act (UE 2024/1689)

## 👥 Équipe du Projet
* **Oussama GUEDRI**
* **Bernard DRUI**
* **Patrice BANAIAS**

---

## 📖 Présentation et Objectifs
Ce projet consiste à concevoir un **Assistant Intelligent Multi-Compétences** capable de naviguer dans le Règlement européen sur l'Intelligence Artificielle. L'objectif est de fournir un système hybride combinant la précision du **RAG** pour les documents internes et la flexibilité des **Agents** pour les actions externes.

L'assistant remplit trois missions clés :
* **Réponse documentée** basée sur un corpus local (IA Act).
* **Exécution d'outils** automatiques selon le contexte (Web, Calculatrice).
* **Conversation contextuelle** avec maintien de l'historique utilisateur.

---

## 🛠️ Stack Technique

Conformément aux contraintes techniques, le système utilise :

| Composant | Outil | Détail |
| :--- | :--- | :--- |
| **Framework** | LangChain | Gestion du RAG et des Agents |
| **Embeddings** | `paraphrase-multilingual-mpnet-base-v2` | 768 dimensions, support multi-langue |
| **Vector Store** | FAISS | Ingestion et indexation de documents |
| **LLM (Local)** | Qwen 2.5 3B (Ollama) | Exécution locale (confidentialité) |
| **LLM (Cloud)** | Mistral 7B (HuggingFace) | Solution via API pour le déploiement cloud |
| **Interface** | Streamlit | Interface conversationnelle web |

---

## 🤖 Intelligence et Routage
Le système utilise un routage intelligent pour traiter les requêtes selon leur nature :

### 1. Mode DIRECT (0 appel LLM)
Utilise des regex pour renvoyer le texte exact d'un article ou d'un considérant spécifique lorsque la question est purement factuelle.

### 2. Mode RAG
Pour les questions d'analyse, l'assistant interroge l'index FAISS. La réponse générée contient des **citations** précises issues des documents internes.

### 3. Mode AGENTS
L'agent intègre un minimum de trois outils pour répondre aux exigences techniques :
* **Recherche Web :** Via DuckDuckGo pour compléter les documents internes.
* **Calculatrice :** Pour évaluer les délais de mise en conformité.
* **Analyseur de Risque :** Outil logique pour classifier les systèmes d'IA selon le règlement.

---

## 📁 Structure du Projet
```text
gen_project/
├── requirements.txt         # Dépendances (LangChain, FAISS, etc.)
├── chunker.py               # Parsing structurel du AI Act
├── build_index.py           # Construction de l'index vectoriel FAISS
├── app.py                   # Application Streamlit (Routage + Mémoire)
├── data/L-202401689FR.md    # Texte source officiel (UE 2024/1689)
└── .streamlit/secrets.toml  # Clés API (HF_TOKEN) - Non commis

## 🚀 Installation et Démarrage

### 1. Configuratio : 
```bash
# Cloner le projet
git clone [https://github.com/guedri-oussama/genai.git](https://github.com/guedri-oussama/genai.git)
cd genai

# Initialisation
```bash
python build_index.py
ollama pull qwen2.5:3b

# Installer les dépendances
pip install -r requirements.txt

# Lancement 
```bash
streamlit run app.py
