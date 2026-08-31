# LLM Tester

Application locale pour envoyer un message à OpenAI, Mistral ou Gemini, puis
afficher côte à côte :

- le texte de réponse extrait ;
- le payload JSON complet renvoyé par le fournisseur.

Les clés sont lues côté serveur depuis le fichier `.env` à la racine du projet.
Elles ne sont jamais envoyées au navigateur.

## Configuration

```dotenv
MISTRAL_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
# Ou : GEMINI_API_KEY=
```

Les modèles proposés par défaut peuvent être remplacés :

```dotenv
MISTRAL_LLM_TEST_MODEL=mistral-large-latest
OPENAI_LLM_TEST_MODEL=gpt-5.6-sol
GOOGLE_LLM_TEST_MODEL=gemini-3.6-flash
```

Le champ « Modèle » reste éditable, même si l’identifiant n’est pas dans les
suggestions.

L'application utilise les mêmes adaptateurs OpenAI, Mistral et Google que le
backend RAG.

Le champ **System** est facultatif. Lorsqu'il est rempli, il est envoyé avant
le champ **Message**, respectivement avec les rôles `system` et `user`.


## Réglages de génération

- OpenAI : `reasoning_effort` (`none` à `xhigh`) et `verbosity` (`low`,
  `medium`, `high`) sont envoyés lorsqu'ils sont sélectionnés.
- Gemini 2.5 : le réglage **Budget de réflexion** envoie `thinking_budget`.
  Utilise `0` pour désactiver la réflexion ou `-1` pour laisser Gemini choisir.
  Les modèles Gemini 3 utilisent un niveau de réflexion, non pris en charge par
  la version actuelle de l'intégration Google installée.
- Mistral : ces deux paramètres ne sont pas exposés par l'adaptateur utilisé.

## Inférence régionale Mistral

Le sélecteur **Lieu d'inférence** permet de choisir l'endpoint global,
européen ou américain. Les endpoints régionaux traitent l'inférence dans la
géographie sélectionnée et comportent une majoration tarifaire de 10 %.

## Résidence des données OpenAI

Le même sélecteur permet de choisir l'endpoint OpenAI global, européen ou
américain. Les endpoints régionaux nécessitent qu'un projet OpenAI éligible ait
été configuré pour la résidence des données.

Pour OpenAI, le sélecteur **Mode de latence** envoie `service_tier=default` ou
`service_tier=fast` directement avec la requête.

## Lancement

Depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe -m uvicorn utils.app_llm_tester.app:app --host 127.0.0.1 --port 8002 --reload --reload-dir utils/app_llm_tester --reload-dir interface/backend
```

Ouvrir ensuite <http://127.0.0.1:8002/>.
