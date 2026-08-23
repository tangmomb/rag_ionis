# Générateur de commandes

Ouvrir `index.html` dans un navigateur. L’outil fonctionne entièrement en local,
sans serveur ni dépendance.

Il génère des commandes PowerShell à exécuter depuis la racine du projet pour :

- l’ingestion YouTube ;
- les commandes `inspect`, `plan`, `run` et `task` du pipeline ;
- la publication S3 et la synchronisation SQL ;
- le lancement des services locaux ;
- la connexion SSH, la mise à jour Git, la reconstruction des conteneurs et
  l’envoi contrôlé de `.env.production` sur le VPS ;
- la consultation en lecture seule de PostgreSQL du VPS : ouverture de `psql`,
  liste des tables de `data` et nombre de lignes par table ;
- la restauration contrôlée d’un dump local dans PostgreSQL sur le VPS ;
- la construction, la publication et l’inspection des images worker Scaleway ;
- les tests et quelques opérations de maintenance.

La rubrique Maintenance inclut également une sauvegarde PostgreSQL : elle crée
un dump binaire daté et validé dans `utils/backup_sql/`, dossier ignoré par Git.

Les définitions sont alignées sur les parseurs `argparse` présents dans
`pipeline/`. La suite de tests est lancée avec `unittest`, sans dépendance
`pytest`.

Pour `pipeline run`, l’interrupteur « Tout OpenAI en Batch » génère le
raccourci `--batch`. Il couvre la réconciliation, la validation
des speakers et les résumés. Les embeddings se lancent séparément avec la tâche
`embeddings.create`.
