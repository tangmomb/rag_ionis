# Database browser

Petite interface locale en lecture seule pour consulter PostgreSQL.

Depuis la racine du projet :

```powershell
python -m uvicorn utils.database_browser.app:app --reload --port 8001
```

Puis ouvrir <http://127.0.0.1:8001>.

L'application lit `DATABASE_URL` dans le fichier `.env`, liste les tables applicatives (y compris le schéma `chat`), affiche les données par pages et permet une recherche dans les colonnes texte. Elle ne propose aucune opération `INSERT`, `UPDATE` ou `DELETE`.
