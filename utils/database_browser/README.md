# Database browser

Interface locale pour explorer les sorties vidéo du pipeline et consulter PostgreSQL.

Depuis la racine du projet :

```powershell
.\run_data_explorer.bat
```

Puis ouvrir <http://127.0.0.1:8003/videos>.

- Explorateur vidéo : <http://127.0.0.1:8003/videos>
- Tables PostgreSQL : <http://127.0.0.1:8003/>

L'explorateur vidéo indexe automatiquement `downloads/youtube/*_init`, puis permet d'ouvrir chaque vidéo et de consulter son résumé, son transcript, les textes OCR, les images extraites, les chunks et les fichiers produits.

L'application lit `DATABASE_URL` dans le fichier `.env`, liste les tables applicatives (y compris le schéma `chat`), affiche les données par pages et permet une recherche dans les colonnes texte. Elle ne propose aucune opération `INSERT`, `UPDATE` ou `DELETE`.
