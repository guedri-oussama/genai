================================================================================
          ASSISTANT JURIDIQUE INTELLIGENT - REGLEMENT IA (UE 2024/1689)
================================================================================

--------------------------------------------------------------------------------
1. INFORMATIONS GÉNÉRALES
--------------------------------------------------------------------------------
PROJET : Assistant Intelligent Multi-Compétences (RAG + Agents) [cite: 1]
GROUPE : Oussama GUEDRI, Bernard DRUI, Patrice BANAIAS
SUJET  : Assistant juridique sur le Règlement (UE) (IA Act) [cite: 48]

--------------------------------------------------------------------------------
2. PRÉSENTATION DU PROJET
--------------------------------------------------------------------------------
L'objectif est de concevoir un assistant capable de retrouver des informations 
précises dans des documents internes et d'exécuter des actions (calculs, 
recherche web, etc.)[cite: 4]. Le système combine deux approches :
- RAG (Retrieval-Augmented Generation) pour répondre aux questions basées sur
  le corpus documentaire[cite: 6].
- Agents et outils pour enrichir les capacités du système[cite: 7].

--------------------------------------------------------------------------------
3. ARCHITECTURE DU SYSTÈME
--------------------------------------------------------------------------------
Le projet est structuré en quatre parties majeures[cite: 13]:

PARTIE 1 - RAG (RETRIEVAL-AUGMENTED GENERATION)[cite: 14]:
- Ingestion et indexation via vectorisation (ChromaDB)[cite: 16].
- Récupération pertinente selon la requête utilisateur[cite: 17].
- Réponses générées via OpenAI ou Mistral via LangChain[cite: 18].

PARTIE 2 - AJOUT D'OUTILS (AGENTS)[cite: 21]:
- Intégration de minimum trois outils[cite: 43].
- Outils inclus : Calculatrice, Recherche web (Tavily/DuckDuckGo) et 
  calendrier/todo list locale[cite: 23, 25, 26].

PARTIE 3 - INTÉGRATION FINALE[cite: 28]:
- Routage intelligent : si la question concerne les documents, le pipeline RAG 
  répond avec des citations obligatoires[cite: 30].
- Si la tâche l'exige, l'agent appelle automatiquement un outil[cite: 11, 31].
- Maintien d'une conversation contextuelle normale pour les salutations[cite: 12, 32].

PARTIE 4 - INTERFACE ET MÉMOIRE[cite: 36]:
- Ajout d'une mémoire conversationnelle pour le suivi du contexte[cite: 37].
- Interface utilisateur développée via Streamlit ou Chainlit[cite: 19, 38].

--------------------------------------------------------------------------------
4. CONTRAINTES TECHNIQUES [cite: 40]
--------------------------------------------------------------------------------
- Framework : LangChain ou LlamaIndex[cite: 41].
- Modèles LLM : OpenAI, Mistral ou HuggingFace[cite: 42].
- Versionnage : Code disponible sur GitHub[cite: 44].
- Documentation : Architecture et instructions d'exécution détaillées[cite: 45].

--------------------------------------------------------------------------------
5. INSTALLATION ET EXÉCUTION
--------------------------------------------------------------------------------
1. Cloner le dépôt : https://github.com/guedri-oussama/genai [cite: 44]
2. Installer les dépendances : pip install -r requirements.txt [cite: 45]
3. Configurer le fichier .env avec les clés API (OPENAI_API_KEY, TAVILY_API_KEY)
4. Lancer l'ingestion du règlement : python ingest.py
5. Démarrer l'assistant : streamlit run app.py