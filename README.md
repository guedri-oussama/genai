# ⚖️ Assistant Juridique Expert - Règlement IA (UE 2024/1689)

## 👥 Équipe du Projet
* **Oussama GUEDRI**
* **Bernard DRUI**
* **Patrice BANAIAS**

---

## 📖 Présentation
Ce projet est un **Assistant Intelligent Multi-Compétences** conçu pour naviguer dans le cadre complexe du Règlement Européen sur l'Intelligence Artificielle (IA Act). [cite_start]Il permet de retrouver des informations précises dans le texte de loi et d'exécuter des actions d'analyse via des agents autonomes[cite: 56, 57].

## 🏗️ Architecture du Système
L'assistant repose sur une architecture combinant **RAG** et **Agents** :

### 1. Pipeline RAG (Savoir Interne)
* [cite_start]**Ingestion :** Indexation du Règlement (UE) 2024/1689[cite: 68].
* [cite_start]**Vectorisation :** Utilisation de `ChromaDB` et `OpenAIEmbeddings`[cite: 68, 70].
* [cite_start]**Récupération :** Recherche sémantique pour fournir des réponses basées exclusivement sur la loi avec citations[cite: 69, 82].

### 2. Agents & Outils (Capacités d'Action)
[cite_start]L'agent utilise trois outils pour enrichir ses réponses[cite: 74, 95]:
* [cite_start]**Recherche Web :** Via Tavily pour les actualités juridiques récentes[cite: 77, 91].
* [cite_start]**Calculatrice :** Pour les calculs de délais de conformité[cite: 75].
* **Analyseur de Risque :** Outil logique pour classifier les systèmes d'IA selon les critères de l'UE.

### 3. Interface & Mémoire
* [cite_start]**Interface :** Développée avec `Streamlit` pour une expérience conversationnelle fluide[cite: 71, 90].
* [cite_start]**Mémoire :** Suivi du contexte pour maintenir la cohérence des échanges[cite: 89].

---

## 🛠️ Contraintes Techniques
* [cite_start]**Framework :** LangChain[cite: 93].
* [cite_start]**LLM :** OpenAI GPT-4o[cite: 94].
* [cite_start]**Base de données :** ChromaDB[cite: 68].

---

## 🚀 Installation et Démarrage

### 1. Configuration
```bash
# Cloner le projet
git clone [https://github.com/guedri-oussama/genai.git](https://github.com/guedri-oussama/genai.git)
cd genai

# Installer les dépendances
pip install -r requirements.txt