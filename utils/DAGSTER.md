# Prototype Dagster pour RAG IONIS

Ce prototype présente les sorties locales du pipeline comme un graphe d'assets partitionné par identifiant YouTube :

```text
video_source -> extracted_images -> ocr_texts -> plain_transcript -> transcript_chunks -> chunk_embeddings
```

Il est volontairement non destructif : matérialiser les assets catalogue et contrôle les fichiers déjà produits, sans relancer les scripts OCR, OpenAI, embeddings, S3 ou SQL.

## Démarrage

```powershell
py -3.10 -m venv .venv-dagster
.\.venv-dagster\Scripts\python.exe -m pip install -r requirements-dagster.txt
.\run_dagster.bat
```

Ouvrir ensuite <http://127.0.0.1:3000>.

Dans **Assets**, sélectionner le groupe `rag_ionis_video`. Chaque identifiant YouTube est une partition. Le job `cataloguer_video` matérialise les six assets pour la partition choisie et exécute les contrôles de qualité.

Le sensor `discover_video_partitions` détecte toutes les 15 secondes les nouvelles vidéos sous `downloads/youtube/*_init`. Le script de démarrage synchronise également les partitions avant de lancer l'interface.

## Contrôles fournis

- fichier vidéo présent et non vide ;
- au moins une image extraite ;
- OCR non vide ;
- transcript d'au moins 100 caractères ;
- au moins un chunk ;
- exactement un embedding par chunk.

Les matérialisations exposent des compteurs, aperçus Markdown, chemins locaux et un lien vers l'explorateur vidéo sur le port 8003.
