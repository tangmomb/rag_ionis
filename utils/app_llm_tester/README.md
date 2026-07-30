# LLM Tester

Application locale pour envoyer un message à OpenAI, Mistral ou Google, puis
afficher côte à côte :

- le texte de réponse extrait ;
- le payload JSON complet renvoyé par le fournisseur.

Les clés sont lues côté serveur depuis le fichier `.env` à la racine du projet.
Elles ne sont jamais envoyées au navigateur.

## Configuration

```dotenv
OPENAI_API_KEY=
MISTRAL_API_KEY=
GOOGLE_API_KEY=
```

`GEMINI_API_KEY` est aussi accepté comme alias de `GOOGLE_API_KEY`.

Les modèles proposés par défaut peuvent être remplacés :

```dotenv
OPENAI_LLM_TEST_MODEL=gpt-5.6-sol
MISTRAL_LLM_TEST_MODEL=mistral-large-latest
GOOGLE_LLM_TEST_MODEL=gemini-3.6-flash
```

Le champ « Modèle » reste éditable, même si l’identifiant n’est pas dans les
suggestions.

L'application partage son adaptateur de payload avec
`utils/run_phoenix_experiment.py` et le backend RAG. Un message commun est
converti vers l'API Responses d'OpenAI, Chat Completions de Mistral ou
`generateContent` de Google, puis normalisé en texte et JSON brut.

## Lancement

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe -m uvicorn utils.app_llm_tester.app:app --host 127.0.0.1 --port 8002 --reload --reload-dir utils/app_llm_tester
```

Ouvrir ensuite <http://127.0.0.1:8002/>.
