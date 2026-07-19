# Générateur de commandes

Ouvrir `index.html` dans un navigateur. L’outil fonctionne entièrement en local,
sans serveur ni dépendance.

Il génère des commandes PowerShell à exécuter depuis la racine du projet pour :

- l’ingestion YouTube ;
- les commandes `inspect`, `plan`, `run` et `task` du pipeline ;
- la publication S3 et la synchronisation SQL ;
- le lancement des services locaux ;
- les tests et quelques opérations de maintenance.

Les définitions sont alignées sur les parseurs `argparse` présents dans
`pipeline/` au moment de la création de l’outil.
