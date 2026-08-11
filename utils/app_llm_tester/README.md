# LLM Tester

Application locale pour envoyer un message à Mistral, puis
afficher côte à côte :

- le texte de réponse extrait ;
- le payload JSON complet renvoyé par le fournisseur.

Les clés sont lues côté serveur depuis le fichier `.env` à la racine du projet.
Elles ne sont jamais envoyées au navigateur.

## Configuration

```dotenv
MISTRAL_API_KEY=
```

Les modèles proposés par défaut peuvent être remplacés :

```dotenv
MISTRAL_LLM_TEST_MODEL=mistral-large-latest
```

Le champ « Modèle » reste éditable, même si l’identifiant n’est pas dans les
suggestions.

L'application utilise le même adaptateur Mistral que le backend RAG. Les tests
OpenAI et Google sont réservés à `utils/run_phoenix_experiment.py`.

## Lancement

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe -m uvicorn utils.app_llm_tester.app:app --host 127.0.0.1 --port 8002 --reload --reload-dir utils/app_llm_tester
```

Ouvrir ensuite <http://127.0.0.1:8002/>.
