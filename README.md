# ⚖️ Assistant Juridique Expert - Règlement IA (UE 2024/1689)

## 👥 Équipe du Projet
* **Oussama GUEDRI**
* **Bernard DRUI**
* **Patrice BANAIAS**

---

## 📖 Présentation
Ce projet est un **Assistant Intelligent Multi-Compétences** conçu pour naviguer dans le cadre complexe du Règlement Européen sur l'Intelligence Artificielle (IA Act). Il permet de retrouver des informations précises dans le texte de loi et d'exécuter des actions d'analyse via des agents autonomes

## 🏗️ Architecture du Système
L'assistant repose sur une architecture combinant **RAG** et **Agents** :

### 1. Pipeline RAG (Savoir Interne)
* **Ingestion :** Indexation du Règlement (UE) 2024/1689
* **Vectorisation :** Utilisation de `ChromaDB` et `OpenAIEmbeddings`
* **Récupération :** Recherche sémantique pour fournir des réponses basées exclusivement sur la loi avec citations

### 2. Agents & Outils (Capacités d'Action)
L'agent utilise trois outils pour enrichir ses réponses:
* **Recherche Web :** Via Tavily pour les actualités juridiques récentes
* **Calculatrice :** Pour les calculs de délais de conformité
* **Analyseur de Risque :** Outil logique pour classifier les systèmes d'IA selon les critères de l'UE.

### 3. Interface & Mémoire
* **Interface :** Développée avec `Streamlit` pour une expérience conversationnelle fluide
* **Mémoire :** Suivi du contexte pour maintenir la cohérence des échanges

---

## 🛠️ Contraintes Techniques
* **Framework :** LangChain
* **LLM :** OpenAI GPT-4o
* **Base de données :** ChromaDB

---

## 🚀 Installation et Démarrage

### 1. Configuration
```bash
# Cloner le projet
git clone [https://github.com/guedri-oussama/genai.git](https://github.com/guedri-oussama/genai.git)
cd genai

# Installer les dépendances
pip install -r requirements.txt