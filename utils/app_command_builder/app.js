const TASKS = [
  ["frames.extract", "Extraire les frames"],
  ["frames.classify", "Classifier les frames"],
  ["video.detect_interview", "Détecter les interviews"],
  ["video.infer_type", "Inférer le type de vidéo"],
  ["ocr.extract_raw", "Extraire l’OCR brut"],
  ["ocr.extract_boxes", "Extraire les positions OCR"],
  ["video.detect_subtitles", "Détecter les sous-titres incrustés"],
  ["ocr.build_processed", "Construire l’OCR traité"],
  ["ocr.filter_overlays", "Filtrer les overlays OCR"],
  ["transcript.extract_ocr", "Construire le transcript OCR"],
  ["transcript.normalize_brand", "Normaliser Ionis-STM"],
  ["transcript.whisper", "Transcrire l’audio avec WhisperX"],
  ["speakers.propose", "Proposer les speakers"],
  ["speakers.validate", "Valider les speakers"],
  ["transcript.correct_whisper", "Corriger WhisperX avec l’OCR visuel"],
  ["transcript.reconcile_ocr", "Corriger WhisperX via les sous-titres OCR"],
  ["transcript.enrich", "Appliquer les speakers et ajouter les intercalaires"],
  ["transcript.create_plain", "Créer le transcript sans timecodes"],
  ["transcript.create_plain_ocr", "Créer l’OCR plain utilisé pour les corrections"],
  ["chunks.create", "Créer les chunks détail"],
  ["chunks.summarize_sections", "Créer les résumés de sections"],
  ["chunks.summarize_video", "Créer le résumé global"],
  ["embeddings.create", "Créer les embeddings"],
];

const selectorField = {
  id: "selector",
  label: "Vidéo(s) à traiter",
  type: "text",
  value: "all",
  placeholder: "all, 5, ID YouTube ou chemin",
  help: "« all », un nombre, un ID YouTube, un fichier vidéo ou un dossier.",
  positional: true,
  required: true,
  full: true,
};

const commonPipelineFields = [
  selectorField,
  {
    id: "root",
    label: "Racine des vidéos",
    flag: "--root",
    type: "text",
    value: "downloads/youtube/init",
    defaultValue: "downloads/youtube/init",
    help: "Ajouté seulement si tu modifies le dossier par défaut.",
    full: true,
  },
  { id: "force", label: "Forcer la régénération", flag: "--force", type: "boolean", help: "Avec run, supprime outputs/ puis reconstruit jusqu’aux chunks ; les embeddings doivent être relancés séparément. Avec task, force seulement la tâche." },
  { id: "dryRun", label: "Simulation", flag: "--dry-run", type: "boolean", help: "Affiche ce qui serait exécuté." },
  {
    id: "openaiMode",
    label: "Mode OpenAI",
    flag: "--openai-mode",
    type: "select",
    value: "normal",
    defaultValue: "normal",
    options: [["normal", "Normal (défaut)"], ["batch", "Batch"]],
    help: "En Batch, tous les appels OpenAI de l’ingestion utilisent l’API Batch.",
  },
  {
    id: "correctionMode",
    label: "Correction Whisper",
    flag: "--correction-mode",
    type: "select",
    value: "balanced",
    defaultValue: "balanced",
    options: [["conservative", "Conservative"], ["balanced", "Balanced (défaut)"], ["aggressive", "Aggressive"]],
  },
  { id: "frameInterval", label: "Intervalle des frames", flag: "--frame-interval", type: "number", value: "0.5", defaultValue: "0.5", min: "0.01", step: "0.1", suffix: "secondes" },
  { id: "detailsPerSection", label: "Détails par section", flag: "--details-per-section", type: "number", value: "6", defaultValue: "6", min: "1", step: "1", suffix: "chunks" },
  { id: "speakerModel", label: "Modèle de validation speakers", flag: "--speaker-validation-model", type: "text", placeholder: "Laisser vide = .env", advanced: true },
  { id: "chunkModel", label: "Modèle de résumé des chunks", flag: "--chunk-summary-model", type: "text", placeholder: "Laisser vide = .env", advanced: true },
];

const runBatchField = {
  id: "batch",
  label: "Tout OpenAI en Batch",
  flag: "--batch",
  type: "boolean",
  help: "Réconciliation, speakers et résumés. Chaque étape attend son résultat avant de poursuivre.",
  full: true,
};

const scalewayImageFields = [
  {
    id: "scalewayRepository",
    label: "Dépôt Container Registry",
    type: "text",
    value: "rg.fr-par.scw.cloud/rag-ionis/rag-ionis-scaleway",
    required: true,
    full: true,
    help: "Namespace Scaleway par défaut : rag-ionis.",
  },
  {
    id: "scalewayCommitTag",
    label: "Tag du commit",
    type: "text",
    placeholder: "Vide = git rev-parse --short=7 HEAD (ex. dd062e7)",
    full: true,
  },
];

const scalewayWorkerConnectionFields = [
  {
    id: "scalewayWorkerHost",
    label: "Hôte SSH de la VM",
    type: "text",
    value: "51.159.135.156",
    required: true,
    full: true,
  },
  {
    id: "scalewayWorkerUser",
    label: "Utilisateur SSH",
    type: "text",
    value: "root",
    required: true,
  },
  {
    id: "scalewayWorkerIdentityFile",
    label: "Clé privée SSH",
    type: "text",
    placeholder: "Optionnelle : utilisée seulement si nécessaire",
    full: true,
  },
];

const VPS_SSH = String.raw`ssh -i "C:\Users\rgb\.ssh\rag_ionis_infomaniak_ed25519" ubuntu@179.237.98.117`;

const actions = [
  {
    id: "fetch",
    category: "YouTube",
    title: "Récupérer les infos YouTube",
    short: "Rafraîchir les métadonnées et commentaires JSON",
    description: "Interroge l’API YouTube, recrée les caches JSON des métadonnées et commentaires, puis les copie dans chaque dossier vidéo local correspondant, sans toucher à PostgreSQL ni retélécharger les vidéos.",
    icon: "↯",
    accent: "#dff0e5",
    module: "pipeline.ingest.fetch_youtube_metadata",
    sections: [
      {
        title: "Sélection",
        fields: [
          { id: "videoUrl", label: "URL d’une vidéo", flag: "--video-url", type: "url", placeholder: "https://www.youtube.com/watch?v=…", help: "Laisse vide pour traiter la chaîne IONIS-STM.", full: true, exclusive: "youtubeSelection" },
          { id: "limit", label: "Limiter le nombre de vidéos", flag: "--limit", type: "number", min: "1", step: "1", placeholder: "Toutes", exclusive: "youtubeSelection" },
        ],
      },
      {
        title: "Stockage",
        fields: [
          { id: "downloadDir", label: "Dossier parent", flag: "--download-dir", type: "text", value: "downloads/youtube", defaultValue: "downloads/youtube", full: true },
          { id: "skipComments", label: "Ne pas récupérer les commentaires", flag: "--skip-comments", type: "boolean", help: "Met uniquement à jour le cache JSON des métadonnées vidéo." },
          { id: "skipTranscripts", label: "Compatibilité : ignorer les transcripts", flag: "--skip-transcripts", type: "boolean", help: "Option conservée par le script, sans effet actuel." },
        ],
      },
    ],
  },
  {
    id: "youtube-update-stats",
    category: "YouTube",
    title: "Update statistiques YouTube",
    short: "Mettre à jour les snapshots de statistiques",
    description: "Interroge l’API YouTube et met à jour uniquement la table stats pour les vidéos déjà présentes dans PostgreSQL. Cette commande ne télécharge aucune vidéo et ne démarre pas la VM GPU ; son journal est conservé dans S3.",
    icon: "↻",
    accent: "#d9eee8",
    fixedArgs: ["-m", "pipeline.update_stats"],
    sections: [],
  },
  {
    id: "youtube-update-videos",
    category: "YouTube",
    title: "Update nouvelles vidéos YouTube",
    short: "Comparer YouTube à PostgreSQL et traiter les absentes",
    description: "Interroge l’API YouTube, compare directement les IDs avec la table videos de PostgreSQL et traite uniquement les vidéos absentes. Chaque update est archivé dans S3 : new_videos.json liste les absentes, puis chaque vidéo est envoyée au pipeline GPU Scaleway, enrichie avec ses embeddings et synchronisée en SQL. Une vidéo en échec reste absente de SQL et sera retentée au prochain lancement.",
    icon: "↻",
    accent: "#d9eee8",
    fixedArgs: ["-m", "pipeline.update_videos"],
    sections: [],
  },
  {
    id: "download",
    category: "YouTube",
    title: "Télécharger les vidéos",
    short: "yt-dlp, 720p MP4 et cache local",
    description: "Télécharge une vidéo précise ou celles présentes dans le cache de métadonnées.",
    icon: "↓",
    accent: "#e8efc7",
    module: "pipeline.ingest.download_videos",
    sections: [
      {
        title: "Sélection",
        fields: [
          { id: "videoUrl", label: "URL d’une vidéo", flag: "--video-url", type: "url", placeholder: "https://www.youtube.com/watch?v=…", help: "Laisse vide pour utiliser le cache de métadonnées.", full: true, exclusive: "youtubeSelection" },
          { id: "limit", label: "Limiter le nombre", flag: "--limit", type: "number", min: "1", step: "1", placeholder: "Toutes", exclusive: "youtubeSelection" },
          { id: "downloadDir", label: "Dossier parent", flag: "--download-dir", type: "text", value: "downloads/youtube", defaultValue: "downloads/youtube", full: true },
        ],
      },
      {
        title: "Téléchargement",
        fields: [
          { id: "cookies", label: "Cookies du navigateur", flag: "--cookies-from-browser", type: "select", value: "", options: [["", "Aucun"], ["brave", "Brave"], ["chrome", "Chrome"], ["chromium", "Chromium"], ["edge", "Edge"], ["firefox", "Firefox"], ["opera", "Opera"], ["vivaldi", "Vivaldi"]] },
          { id: "force", label: "Retélécharger", flag: "--force", type: "boolean", help: "Remplace les fichiers déjà présents." },
          { id: "minDelay", label: "Délai minimum", flag: "--min-delay", type: "number", value: "15", defaultValue: "15", min: "0", step: "1", suffix: "secondes" },
          { id: "maxDelay", label: "Délai maximum", flag: "--max-delay", type: "number", value: "45", defaultValue: "45", min: "0", step: "1", suffix: "secondes" },
          { id: "dryRun", label: "Simulation", flag: "--dry-run", type: "boolean", help: "Liste les vidéos sans les télécharger." },
        ],
      },
    ],
  },
  {
    id: "inspect",
    category: "Pipeline",
    title: "Inspecter une vidéo",
    short: "Probe, détection visuelle et OCR",
    description: "Sonde la vidéo et exécute les détecteurs nécessaires au routage.",
    icon: "⌕",
    accent: "#dbeee9",
    pipelineCommand: "inspect",
    extraFields: [{ id: "probeOnly", label: "Probe technique seulement", flag: "--probe-only", type: "boolean", help: "N’exécute aucun traitement visuel ou OCR." }],
  },
  {
    id: "plan",
    category: "Pipeline",
    title: "Afficher le plan",
    short: "Prévisualiser les étapes sans les exécuter",
    description: "Recalcule le plan adapté à la vidéo, sans produire de sorties métier.",
    icon: "≋",
    accent: "#f3e7c1",
    pipelineCommand: "plan",
    extraFields: [{ id: "includeInspection", label: "Inclure l’inspection", flag: "--include-inspection", type: "boolean", help: "Affiche aussi le plan d’inspection." }],
  },
  {
    id: "run",
    category: "Pipeline",
    title: "Lancer le pipeline",
    short: "Traitement jusqu’aux chunks, en direct ou OpenAI Batch",
    description: "Inspecte puis exécute la chaîne adaptée à chaque vidéo jusqu’à la création des chunks. Les embeddings se lancent séparément avec la tâche dédiée.",
    icon: "▶",
    accent: "#d9f36d",
    pipelineCommand: "run",
    extraFields: [{ id: "skipInspection", label: "Réutiliser l’inspection", flag: "--skip-inspection", type: "boolean", help: "Réutilise les faits déjà présents dans le manifeste." }],
  },
  {
    id: "task",
    category: "Pipeline",
    title: "Exécuter une tâche précise",
    short: "Une étape exacte du catalogue",
    description: "Exécute une seule tâche du registre sur la sélection de vidéos.",
    icon: "◆",
    accent: "#eddff0",
    pipelineCommand: "task",
    task: true,
  },
  {
    id: "task-list",
    category: "Pipeline",
    title: "Lister les tâches",
    short: "Afficher le catalogue disponible",
    description: "Affiche les identifiants, phases et titres de toutes les tâches.",
    icon: "☷",
    accent: "#e3e8ef",
    fixedArgs: ["-m", "pipeline", "task", "--list"],
    sections: [],
  },
  {
    id: "upload-s3",
    category: "Publication",
    title: "Uploader vers S3",
    short: "Publier les sorties en conservant l’arborescence",
    description: "Envoie un dossier de vidéos et ses sorties vers le bucket configuré.",
    icon: "↑",
    accent: "#dce8f5",
    module: "pipeline.publish.upload_outputs_to_s3",
    sections: [
      {
        title: "Source et destination",
        fields: [
          { id: "videoDir", label: "Dossier à uploader", flag: "--video-dir", type: "text", placeholder: "Défaut : dernier dossier", full: true },
          { id: "downloadDir", label: "Dossier parent", flag: "--download-dir", type: "text", value: "downloads/youtube", defaultValue: "downloads/youtube" },
          { id: "bucket", label: "Bucket S3", flag: "--bucket", type: "text", placeholder: "Défaut : S3_BUCKET_NAME" },
          { id: "region", label: "Région AWS", flag: "--region", type: "text", placeholder: "Défaut : S3_REGION" },
          { id: "prefix", label: "Préfixe S3", flag: "--prefix", type: "text", placeholder: "youtube/…", exclusive: "prefixMode" },
          { id: "noPrefix", label: "Uploader à la racine", flag: "--no-prefix", type: "boolean", help: "Incompatible avec un préfixe explicite.", exclusive: "prefixMode" },
          { id: "videoIds", label: "Limiter à des IDs vidéo", flag: "--video-id", type: "textarea", placeholder: "Un ID par ligne", help: "L’option --video-id sera répétée.", repeatable: true, full: true },
        ],
      },
      {
        title: "Comportement",
        fields: [
          { id: "force", label: "Écraser les objets existants", flag: "--force", type: "boolean" },
          { id: "dryRun", label: "Simulation", flag: "--dry-run", type: "boolean" },
          { id: "cleanInit", label: "Nettoyer les anciens préfixes", flag: "--clean-init-prefix", type: "boolean", help: "Supprime les anciens préfixes avant upload." },
        ],
      },
    ],
    warning: "Avec « Nettoyer les anciens préfixes » sans simulation, des objets S3 peuvent être supprimés.",
  },
  {
    id: "sync-db",
    category: "Publication",
    title: "Synchroniser la base SQL",
    short: "Vidéos, stats, commentaires, transcripts et chunks",
    description: "Met à jour PostgreSQL à partir des fichiers locaux, dont metadata/youtube_comments.json, et publie uniquement la chaîne de transcript WhisperX canonique.",
    icon: "⇄",
    accent: "#e7e1f4",
    module: "pipeline.publish.sync_database",
    sections: [
      {
        title: "Source",
        fields: [
          { id: "videoDir", label: "Dossier traité", flag: "--video-dir", type: "text", placeholder: "Défaut : dernier dossier", full: true },
          { id: "downloadDir", label: "Dossier parent", flag: "--download-dir", type: "text", value: "downloads/youtube", defaultValue: "downloads/youtube" },
          { id: "videoIds", label: "Limiter à des IDs vidéo", flag: "--video-id", type: "textarea", placeholder: "Un ID par ligne", repeatable: true, full: true },
        ],
      },
      {
        title: "S3 et base",
        fields: [
          { id: "bucket", label: "Bucket S3", flag: "--bucket", type: "text", placeholder: "Défaut : .env" },
          { id: "prefix", label: "Préfixe S3", flag: "--prefix", type: "text", placeholder: "youtube/…", exclusive: "prefixMode" },
          { id: "noPrefix", label: "Fichiers à la racine du bucket", flag: "--no-prefix", type: "boolean", exclusive: "prefixMode" },
          { id: "skipTranscripts", label: "Ignorer transcripts et chunks", flag: "--skip-transcripts", type: "boolean" },
          { id: "skipComments", label: "Ignorer les commentaires", flag: "--skip-comments", type: "boolean" },
          { id: "dryRun", label: "Simulation", flag: "--dry-run", type: "boolean" },
          { id: "resetDatabase", label: "Recréer le schéma data", flag: "--reset-database", type: "boolean", help: "Supprime et recrée les données avant la synchronisation." },
          { id: "cleanInit", label: "Option legacy clean-init-assets", flag: "--clean-init-assets", type: "boolean", help: "Conservée pour compatibilité, sans effet." },
        ],
      },
    ],
    warning: "« Recréer le schéma data » supprime les données du schéma avant de les réimporter.",
  },
  {
    id: "api",
    category: "Services",
    title: "Démarrer l’API RAG",
    short: "FastAPI avec rechargement automatique",
    description: "Lance l’interface backend locale avec Uvicorn.",
    icon: "◉",
    accent: "#dcefe3",
    module: "uvicorn",
    moduleTarget: "interface.app:app",
    sections: [
      {
        title: "Serveur",
        fields: [
          { id: "host", label: "Hôte", flag: "--host", type: "text", value: "127.0.0.1", required: true },
          { id: "port", label: "Port", flag: "--port", type: "number", value: "8006", required: true, min: "1", max: "65535" },
          { id: "reload", label: "Rechargement automatique", flag: "--reload", type: "boolean", checked: true },
          { id: "reloadDir", label: "Dossier surveillé", flag: "--reload-dir", type: "text", value: "interface" },
        ],
      },
    ],
  },
  {
    id: "db-browser",
    category: "Services",
    title: "Démarrer le navigateur SQL",
    short: "Interface locale de consultation de la base",
    description: "Lance l’utilitaire Database Browser avec Uvicorn.",
    icon: "▦",
    accent: "#f0e6d7",
    module: "uvicorn",
    moduleTarget: "utils.app_database_browser.app:app",
    sections: [
      {
        title: "Serveur",
        fields: [
          { id: "host", label: "Hôte", flag: "--host", type: "text", value: "127.0.0.1", required: true },
          { id: "port", label: "Port", flag: "--port", type: "number", value: "8001", required: true, min: "1", max: "65535" },
          { id: "reload", label: "Rechargement automatique", flag: "--reload", type: "boolean", checked: true },
          { id: "reloadDir", label: "Dossier surveillé", flag: "--reload-dir", type: "text", value: "utils/app_database_browser" },
        ],
      },
    ],
  },
  {
    id: "speakers-merge",
    category: "Maintenance",
    title: "Fusionner les speakers en doublon",
    short: "Rapprocher les noms et choisir le poste à conserver",
    description: "Analyse uniquement les noms de la table speakers, propose les doublons identiques ou très proches, puis ouvre une fenêtre Tkinter pour choisir le nom et sélectionner ou saisir le poste à conserver.",
    icon: "⇉",
    accent: "#dce8f5",
    fixedArgs: ["-m", "pipeline", "speakers-merge"],
    sections: [
      {
        title: "Détection et exécution",
        fields: [
          {
            id: "maxDistance",
            label: "Tolérance orthographique",
            flag: "--max-distance",
            type: "select",
            value: "2",
            defaultValue: "2",
            options: [
              ["2", "Jusqu’à 2 lettres (défaut)"],
              ["1", "Jusqu’à 1 lettre"],
              ["0", "Noms normalisés identiques"],
            ],
            help: "Compte les insertions, suppressions ou substitutions après normalisation de la casse, des accents et de la ponctuation.",
            full: true,
          },
          {
            id: "dryRun",
            label: "Afficher seulement les propositions",
            flag: "--dry-run",
            type: "boolean",
            help: "N’ouvre aucun choix interactif et ne modifie pas PostgreSQL.",
            full: true,
          },
        ],
      },
    ],
    warning: "Sans simulation, chaque fusion confirmée harmonise le nom et le poste dans PostgreSQL et peut supprimer une ligne en doublon pour une même vidéo.",
  },
  {
    id: "tests",
    category: "Maintenance",
    title: "Lancer les tests",
    short: "Découverte de la suite unittest",
    description: "Exécute les tests standards du projet sans dépendance pytest.",
    icon: "✓",
    accent: "#e1eddc",
    fixedArgs: ["-m", "unittest", "discover"],
    sections: [
      {
        title: "Découverte",
        fields: [
          { id: "startDirectory", label: "Dossier de tests", flag: "-s", type: "text", value: "tests", full: true },
          { id: "pattern", label: "Motif de fichiers", flag: "-p", type: "text", value: "test_*.py", placeholder: "test_*.py" },
          { id: "topDirectory", label: "Racine d’import", flag: "-t", type: "text", placeholder: "Défaut : dossier de tests" },
          { id: "verbose", label: "Sortie détaillée", flag: "-v", type: "boolean" },
          { id: "failFast", label: "Arrêter au premier échec", flag: "-f", type: "boolean" },
          { id: "keyword", label: "Filtre par nom", flag: "-k", type: "text", placeholder: "pipeline_execution" },
        ],
      },
    ],
  },
  {
    id: "clean-init-generated",
    category: "Maintenance",
    title: "Garder uniquement les vidéos d’init",
    short: "Supprimer tous les dossiers metadata/ et outputs/",
    description: "Supprime les données générées de chaque dossier vidéo dans init, sans supprimer les vidéos ni les autres fichiers.",
    icon: "⌫",
    accent: "#f5ddd5",
    fixedArgs: ["utils/clean_init_generated.py"],
    sections: [
      {
        title: "Nettoyage",
        fields: [
          {
            id: "root",
            label: "Racine des vidéos",
            flag: "--root",
            type: "text",
            value: "downloads/youtube/init",
            defaultValue: "downloads/youtube/init",
            help: "Chaque sous-dossier direct est traité comme un dossier vidéo.",
            full: true,
          },
          {
            id: "dryRun",
            label: "Simulation",
            flag: "--dry-run",
            type: "boolean",
            help: "Liste les dossiers ciblés sans rien supprimer.",
          },
        ],
      },
    ],
    warning: "Sans simulation, tous les dossiers metadata/ et outputs/ sous init sont supprimés définitivement. Les fichiers vidéo sont conservés.",
  },
  {
    id: "clear-db",
    category: "Maintenance",
    title: "Vider les tables SQL",
    short: "TRUNCATE avec remise à zéro des identifiants",
    description: "Vide les tables applicatives sans supprimer leur structure.",
    icon: "⌫",
    accent: "#f5ddd5",
    fixedArgs: ["utils/clear_database.py"],
    sections: [],
    warning: "Cette commande vide les données SQL. Elle ne propose pas de mode simulation.",
  },
  {
    id: "reset-db",
    category: "Maintenance",
    title: "Recréer la base SQL",
    short: "Réinitialiser le schéma depuis le projet",
    description: "Supprime puis recrée la structure de données attendue.",
    icon: "↻",
    accent: "#f4d8d1",
    fixedArgs: ["utils/reset_database.py"],
    sections: [],
    warning: "Cette commande recrée la base et peut supprimer les données existantes.",
  },
  {
    id: "migrate-embeddings",
    category: "Maintenance",
    title: "Migrer les embeddings",
    short: "Passer les vecteurs de 3072 à 2000 dimensions",
    description: "Migre les embeddings SQL par lots et reconstruit l’index vectoriel.",
    icon: "◇",
    accent: "#e5e0f0",
    fixedArgs: ["utils/migrate_embeddings_2000.py"],
    sections: [
      {
        title: "Migration",
        fields: [
          { id: "batchSize", label: "Taille des lots", flag: "--batch-size", type: "number", value: "32", defaultValue: "32", min: "1", step: "1" },
          { id: "dryRun", label: "Inspecter sans modifier", flag: "--dry-run", type: "boolean" },
        ],
      },
    ],
  },
  {
    id: "vps-connect",
    category: "VPS",
    title: "Se connecter au VPS",
    short: "Ouvrir une session SSH interactive",
    description: "Ouvre un terminal SSH interactif dans le dossier personnel de l’utilisateur du VPS.",
    icon: "⌁",
    accent: "#d9f36d",
    shellCommand: VPS_SSH,
    sections: [],
  },
  {
    id: "vps-pull-code",
    category: "VPS",
    title: "Mettre à jour le code du VPS",
    short: "Récupérer le dernier commit avec git pull",
    description: "À lancer depuis une session SSH déjà ouverte : exécute un git pull --ff-only dans le dépôt du VPS. La commande s’arrête si des modifications locales empêchent une mise à jour propre.",
    icon: "↓",
    accent: "#dce8f5",
    shellCommand: "cd /rag_ionis && git pull --ff-only",
    shellLabel: "Commande Linux VPS",
    terminal: { label: "Linux · VPS", prompt: "ubuntu@vps:~$" },
    sections: [],
  },
  {
    id: "vps-rebuild-stack",
    category: "VPS",
    title: "Reconstruire le stack VPS",
    short: "Reconstruire les images et redémarrer les services",
    description: "À lancer depuis une session SSH déjà ouverte : reconstruit les images de production puis redémarre le stack avec les derniers fichiers déployés.",
    icon: "↻",
    accent: "#e7e1f4",
    shellCommand: "cd /rag_ionis && docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.production up -d --build",
    shellLabel: "Commande Linux VPS",
    terminal: { label: "Linux · VPS", prompt: "ubuntu@vps:~$" },
    sections: [],
  },
  {
    id: "vps-send-production-env",
    category: "VPS",
    title: "Renvoyer .env.production",
    short: "Copier l’environnement local vers le VPS",
    description: "Envoie le fichier .env.production local vers /srv/rag_ionis/.env.production via SCP.",
    icon: "↑",
    accent: "#f4d8d1",
    shellCommand: String.raw`scp -i "C:\Users\rgb\.ssh\rag_ionis_infomaniak_ed25519" ".env.production" ubuntu@179.237.98.117:/srv/rag_ionis/.env.production`,
    sections: [],
    warning: "Cette commande transmet les secrets de .env.production au VPS et remplace le fichier distant. À utiliser uniquement si nécessaire.",
  },
  {
    id: "scaleway-build-image",
    category: "Scaleway",
    title: "Construire l’image Scaleway",
    short: "Créer les tags commit et latest sans pousser",
    description: "Construit localement l’image GPU linux/amd64 avec le tag court du commit et latest, sans l’envoyer au registre.",
    icon: "◆",
    accent: "#d9f36d",
    commandBuilder: buildScalewayLocalBuildCommand,
    sections: [{ title: "Image", fields: scalewayImageFields }],
  },
  {
    id: "scaleway-publish-image",
    category: "Scaleway",
    title: "Publier l’image Scaleway",
    short: "Envoyer le commit et latest au registre",
    description: "Construit puis pousse la même image worker sous le tag du commit et sous latest avec le script lancé via Git Bash.",
    icon: "↑",
    accent: "#dce8f5",
    commandBuilder: buildScalewayPublishCommand,
    sections: [{ title: "Image", fields: scalewayImageFields }],
    warning: "Vérifie le namespace, le login Docker et le commit avant de pousser. Le tag latest sera déplacé vers cette image.",
  },
  {
    id: "scaleway-inspect-image",
    category: "Scaleway",
    title: "Inspecter une image Scaleway",
    short: "Afficher le digest et les métadonnées du registre",
    description: "Interroge le Container Registry Scaleway avec docker buildx imagetools inspect.",
    icon: "⌕",
    accent: "#e7e1f4",
    commandBuilder: buildScalewayInspectCommand,
    sections: [
      {
        title: "Image distante",
        fields: [
          { ...scalewayImageFields[0] },
          { id: "scalewayInspectTag", label: "Tag à inspecter", type: "text", value: "latest", defaultValue: "latest", required: true },
        ],
      },
    ],
  },
  {
    id: "scaleway-pull-latest",
    category: "Scaleway",
    title: "Récupérer latest localement",
    short: "Tester le pull de l’image publiée",
    description: "Télécharge le tag latest depuis le Container Registry pour vérifier l’accès et la publication.",
    icon: "↓",
    accent: "#dff0e5",
    commandBuilder: () => ({
      command: `docker pull ${quotePowerShell(`${scalewayRepository()}:latest`)}`,
      summary: "Pull local de latest",
    }),
    sections: [{ title: "Image", fields: [scalewayImageFields[0]] }],
  },
  {
    id: "scaleway-worker-info",
    category: "Scaleway",
    title: "Vérifier la configuration worker",
    short: "Lire le service et l’environnement actifs sur la VM",
    description: "Se connecte à la VM Scaleway indiquée ci-dessous pour afficher l’unité systemd active et le fichier d’environnement réellement utilisé.",
    icon: "☷",
    accent: "#f4d8d1",
    commandBuilder: buildScalewayWorkerInfoCommand,
    sections: [{ title: "Connexion à la VM", fields: scalewayWorkerConnectionFields }],
    warning: "La VM Scaleway doit être démarrée et accessible en SSH. Cette commande affiche les clés S3 et API en clair.",
  },
];

const categories = ["Toutes", "Pipeline", "YouTube", "Publication", "Services", "Maintenance", "VPS", "Scaleway"];
const launchers = {
  venv: String.raw`.\.venv\Scripts\python.exe`,
  python: "python",
  py: "py",
};

const state = {
  actionId: "run",
  category: "Toutes",
  search: "",
  wrapped: false,
};

const elements = {
  actionList: document.querySelector("#action-list"),
  categoryTabs: document.querySelector("#category-tabs"),
  search: document.querySelector("#action-search"),
  form: document.querySelector("#options-form"),
  launcherRow: document.querySelector("#launcher-row"),
  launcher: document.querySelector("#launcher"),
  title: document.querySelector("#selected-title"),
  description: document.querySelector("#selected-description"),
  icon: document.querySelector("#selected-icon"),
  badge: document.querySelector("#action-badge"),
  output: document.querySelector("#command-output"),
  summary: document.querySelector("#option-summary"),
  terminalLabel: document.querySelector("#terminal-label"),
  terminalPrompt: document.querySelector("#terminal-prompt"),
  warning: document.querySelector("#warning-box"),
  copy: document.querySelector("#copy-button"),
  wrap: document.querySelector("#wrap-button"),
  reset: document.querySelector("#reset-button"),
  toast: document.querySelector("#toast"),
};

function currentAction() {
  return actions.find((action) => action.id === state.actionId);
}

function pipelineSections(action) {
  const primaryFields = [];
  if (action.task) {
    primaryFields.push({
      id: "taskId",
      label: "Tâche",
      type: "select",
      positional: true,
      required: true,
      value: TASKS[0][0],
      options: TASKS.map(([id, title]) => [id, `${id} — ${title}`]),
      full: true,
    });
  }
  primaryFields.push(...commonPipelineFields.slice(0, 2));
  if (action.extraFields) primaryFields.push(...action.extraFields);
  const executionFields = commonPipelineFields.slice(2, 9);
  if (action.id === "run") {
    const openaiModeIndex = executionFields.findIndex(
      (field) => field.id === "openaiMode",
    );
    executionFields.splice(openaiModeIndex, 1, runBatchField);
  }

  return [
    { title: action.task ? "Tâche et sélection" : "Sélection", fields: primaryFields },
    { title: "Exécution", fields: executionFields },
    { title: "Modèles avancés", fields: commonPipelineFields.slice(9) },
  ];
}

function actionSections(action) {
  return action.pipelineCommand ? pipelineSections(action) : (action.sections || []);
}

function renderCategories() {
  elements.categoryTabs.innerHTML = categories.map((category) => (
    `<button class="category-tab${state.category === category ? " active" : ""}" type="button" data-category="${category}">${category}</button>`
  )).join("");
}

function renderActions() {
  const query = state.search.trim().toLocaleLowerCase("fr");
  const filtered = actions.filter((action) => {
    const inCategory = state.category === "Toutes" || action.category === state.category;
    const haystack = `${action.title} ${action.short} ${action.category}`.toLocaleLowerCase("fr");
    return inCategory && (!query || haystack.includes(query));
  });

  elements.actionList.innerHTML = filtered.length
    ? filtered.map((action) => `
      <button class="action-item${action.id === state.actionId ? " active" : ""}" type="button" data-action="${action.id}" style="--accent:${action.accent}">
        <span class="action-icon" aria-hidden="true">${action.icon}</span>
        <span><strong>${action.title}</strong><small>${action.short}</small></span>
        <span class="action-arrow" aria-hidden="true">›</span>
      </button>
    `).join("")
    : `<div class="empty-state">Aucune action ne correspond à cette recherche.</div>`;
}

function renderField(field) {
  if (field.type === "boolean") {
    return `
      <div class="toggle-field${field.full ? " full" : ""}">
        <div>
          <strong>${field.label}</strong>
          ${field.help ? `<small>${field.help}</small>` : ""}
        </div>
        <label class="switch">
          <input id="${field.id}" name="${field.id}" type="checkbox" ${field.checked ? "checked" : ""} data-flag="${field.flag || ""}" data-exclusive="${field.exclusive || ""}">
          <span aria-hidden="true"></span>
        </label>
      </div>`;
  }

  const hint = field.required
    ? `<span class="required-mark">requis</span>`
    : field.suffix
      ? `<span>${field.suffix}</span>`
      : field.defaultValue !== undefined
        ? `<span>défaut : ${field.defaultValue}</span>`
        : "";
  let control;
  if (field.type === "select") {
    control = `<select id="${field.id}" name="${field.id}" data-flag="${field.flag || ""}" data-positional="${Boolean(field.positional)}" data-default="${field.defaultValue ?? ""}" data-exclusive="${field.exclusive || ""}">
      ${field.options.map(([value, label]) => `<option value="${value}" ${String(field.value ?? "") === String(value) ? "selected" : ""}>${label}</option>`).join("")}
    </select>`;
  } else if (field.type === "textarea") {
    control = `<textarea id="${field.id}" name="${field.id}" placeholder="${field.placeholder || ""}" data-flag="${field.flag || ""}" data-repeatable="${Boolean(field.repeatable)}" data-exclusive="${field.exclusive || ""}">${field.value || ""}</textarea>`;
  } else {
    const inputType = ["url", "number", "password"].includes(field.type) ? field.type : "text";
    control = `<input id="${field.id}" name="${field.id}" type="${inputType}" value="${field.value ?? ""}" placeholder="${field.placeholder || ""}" ${field.min !== undefined ? `min="${field.min}"` : ""} ${field.max !== undefined ? `max="${field.max}"` : ""} ${field.step !== undefined ? `step="${field.step}"` : ""} ${field.required ? "required" : ""} data-flag="${field.flag || ""}" data-positional="${Boolean(field.positional)}" data-default="${field.defaultValue ?? ""}" data-exclusive="${field.exclusive || ""}">`;
  }

  return `
    <div class="field${field.full ? " full" : ""}">
      <label for="${field.id}">${field.label}${hint}</label>
      ${control}
      ${field.help ? `<p class="field-help">${field.help}</p>` : ""}
    </div>`;
}

function renderForm() {
  const action = currentAction();
  const sections = actionSections(action);
  elements.title.textContent = action.title;
  elements.description.textContent = action.description;
  elements.icon.textContent = action.icon;
  elements.icon.style.background = action.accent;
  elements.badge.textContent = action.category;
  elements.launcherRow.hidden = Boolean(action.shellCommand || action.commandBuilder);
  elements.form.innerHTML = sections.map((section) => `
    <section class="form-section">
      <h4 class="form-section-title">${section.title}</h4>
      <div class="field-grid">${section.fields.map(renderField).join("")}</div>
    </section>
  `).join("");
  elements.warning.hidden = !action.warning;
  elements.warning.textContent = action.warning ? `Attention — ${action.warning}` : "";
  updateCommand();
}

function quotePowerShell(value) {
  const stringValue = String(value);
  if (/^[a-zA-Z0-9_./:\\=@?&%+-]+$/.test(stringValue)) return stringValue;
  return `'${stringValue.replaceAll("'", "''")}'`;
}

function collectArgs() {
  const args = [];
  const positionals = [];
  let activeOptions = 0;
  const controls = [...elements.form.querySelectorAll("input, select, textarea")];

  for (const control of controls) {
    const flag = control.dataset.flag;
    if (control.type === "checkbox") {
      if (control.checked && flag) {
        args.push(flag);
        activeOptions += 1;
      }
      continue;
    }

    const value = control.value.trim();
    if (!value) continue;
    const isPositional = control.dataset.positional === "true";
    const defaultValue = control.dataset.default;
    if (!isPositional && defaultValue !== undefined && value === defaultValue) continue;

    if (control.dataset.repeatable === "true") {
      const values = value.split(/\r?\n|,/).map((item) => item.trim()).filter(Boolean);
      values.forEach((item) => args.push(flag, quotePowerShell(item)));
      activeOptions += values.length;
    } else if (isPositional) {
      positionals.push(value);
    } else if (flag) {
      args.push(flag, quotePowerShell(value));
      activeOptions += 1;
    }
  }
  if (positionals.some((value) => value.startsWith("-"))) {
    args.push("--");
  }
  args.push(...positionals.map(quotePowerShell));
  return { args, activeOptions };
}

function baseArgs(action) {
  if (action.fixedArgs) return [...action.fixedArgs];
  if (action.pipelineCommand) return ["-m", "pipeline", action.pipelineCommand];
  const args = ["-m", action.module];
  if (action.moduleTarget) args.push(action.moduleTarget);
  return args;
}

function formValue(id, fallback = "") {
  const control = elements.form.elements.namedItem(id);
  return control ? control.value.trim() : fallback;
}

function powerShellString(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

function buildVpsDatabaseCommand() {
  const user = formValue("vpsDatabaseUser", "rag_ionis") || "rag_ionis";
  const password = formValue("vpsDatabasePassword") || "REMPLACER_PAR_POSTGRES_PASSWORD";
  const database = formValue("vpsDatabaseName", "rag_ionis") || "rag_ionis";
  const mode = formValue("vpsDatabaseMode", "dry-run");
  const databaseUrl = `postgresql://${encodeURIComponent(user)}:${encodeURIComponent(password)}@127.0.0.1:15432/${encodeURIComponent(database)}`;
  const setup = [
    `$env:PYTHON_DOTENV_DISABLED = "1"`,
    `$env:DATABASE_URL = ${powerShellString(databaseUrl)}`,
  ];

  if (mode === "test") {
    return {
      command: `${setup.join("\n")}\n${launchers.venv} -c "import os, psycopg; c=psycopg.connect(os.environ['DATABASE_URL']); print(c.execute('SELECT current_database(), current_user').fetchone()); c.close()"`,
      summary: "Test de connexion au VPS",
    };
  }

  const { args } = collectArgs();
  const syncArgs = [launchers.venv, "-m", "pipeline.publish.sync_database", ...args];
  if (mode === "dry-run") syncArgs.push("--dry-run");
  return {
    command: `${setup.join("\n")}\n${syncArgs.join(" ")}`,
    summary: mode === "dry-run" ? "Simulation sur la base du VPS" : "Synchronisation réelle de la base du VPS",
  };
}

function scalewayRepository() {
  return formValue(
    "scalewayRepository",
    "rg.fr-par.scw.cloud/rag-ionis/rag-ionis-scaleway",
  ) || "rg.fr-par.scw.cloud/rag-ionis/rag-ionis-scaleway";
}

function scalewayCommitTag() {
  return formValue("scalewayCommitTag");
}

function buildScalewayPublishCommand() {
  const repository = scalewayRepository();
  const commitTag = scalewayCommitTag();
  const tagArgument = commitTag ? ` ${quotePowerShell(commitTag)}` : "";
  const command = [
    `$env:SCALEWAY_IMAGE_REPOSITORY = ${powerShellString(repository)}`,
    `& "C:\\Program Files\\Git\\bin\\bash.exe" deploy/publish-scaleway-image.sh${tagArgument}`,
  ].join("\n");
  return {
    command,
    summary: "Publication des tags commit et latest",
  };
}

function buildScalewayLocalBuildCommand() {
  const repository = scalewayRepository();
  const commitTag = scalewayCommitTag();
  const tagExpression = commitTag
    ? powerShellString(commitTag)
    : "(git rev-parse --short=7 HEAD).Trim()";
  return {
    command: [
      `$repository = ${powerShellString(repository)}`,
      `$commitTag = ${tagExpression}`,
      'docker build --platform linux/amd64 -f Dockerfile.scaleway -t "${repository}:$commitTag" -t "${repository}:latest" .',
    ].join("\n"),
    summary: "Build local sans publication",
  };
}

function buildScalewayInspectCommand() {
  const repository = scalewayRepository();
  const tag = formValue("scalewayInspectTag", "latest") || "latest";
  return {
    command: `docker buildx imagetools inspect ${quotePowerShell(`${repository}:${tag}`)}`,
    summary: `Inspection du registre : ${tag}`,
  };
}

function buildScalewayWorkerInfoCommand() {
  const host = formValue("scalewayWorkerHost", "51.159.135.156") || "51.159.135.156";
  const user = formValue("scalewayWorkerUser", "root") || "root";
  const identityFile = formValue("scalewayWorkerIdentityFile");
  const identityOption = identityFile ? ` -i ${quotePowerShell(identityFile)}` : "";
  return {
    command: `ssh${identityOption} ${quotePowerShell(`${user}@${host}`)} "sudo systemctl cat rag-ionis-scaleway-worker; printf '\\n--- Environnement worker ---\\n'; sudo cat /etc/rag-ionis/scaleway-worker.env"`,
    summary: "Vérification distante du worker",
  };
}

function setTerminalContext(terminal) {
  const context = terminal || { label: "PowerShell · rag_ionis", prompt: "PS rag_ionis>" };
  elements.terminalLabel.textContent = context.label;
  elements.terminalPrompt.textContent = context.prompt;
}

function updateCommand() {
  const action = currentAction();
  setTerminalContext(action.terminal);
  if (action.shellCommand) {
    elements.output.textContent = action.shellCommand;
    elements.summary.textContent = action.shellLabel || "Commande PowerShell VPS";
    return;
  }
  if (action.commandBuilder) {
    const result = action.commandBuilder();
    elements.output.textContent = result.command;
    elements.summary.textContent = result.summary;
    return;
  }
  const { args, activeOptions } = collectArgs();
  const command = [launchers[elements.launcher.value], ...baseArgs(action), ...args].join(" ");
  elements.output.textContent = command;
  elements.summary.textContent = activeOptions
    ? `${activeOptions} option${activeOptions > 1 ? "s" : ""} personnalisée${activeOptions > 1 ? "s" : ""}`
    : "Commande minimale";
}

function enforceExclusive(changedControl) {
  const group = changedControl.dataset.exclusive;
  if (!group) return;
  const hasValue = changedControl.type === "checkbox" ? changedControl.checked : Boolean(changedControl.value.trim());
  if (!hasValue) return;

  elements.form.querySelectorAll(`[data-exclusive="${group}"]`).forEach((control) => {
    if (control === changedControl) return;
    if (control.type === "checkbox") control.checked = false;
    else control.value = "";
  });
}

function selectAction(actionId) {
  state.actionId = actionId;
  renderActions();
  renderForm();
  if (window.innerWidth < 901) {
    document.querySelector(".config-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function copyCommand() {
  const command = elements.output.textContent;
  try {
    await navigator.clipboard.writeText(command);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(elements.output);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    document.execCommand("copy");
    selection.removeAllRanges();
  }
  elements.copy.innerHTML = `<span aria-hidden="true">✓</span> Copiée`;
  elements.toast.classList.add("visible");
  window.setTimeout(() => {
    elements.copy.innerHTML = `<span aria-hidden="true">▣</span> Copier`;
    elements.toast.classList.remove("visible");
  }, 1500);
}

elements.categoryTabs.addEventListener("click", (event) => {
  const button = event.target.closest("[data-category]");
  if (!button) return;
  state.category = button.dataset.category;
  renderCategories();
  renderActions();
});

elements.actionList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-action]");
  if (button) selectAction(button.dataset.action);
});

elements.search.addEventListener("input", () => {
  state.search = elements.search.value;
  renderActions();
});

elements.form.addEventListener("input", (event) => {
  enforceExclusive(event.target);
  updateCommand();
});
elements.form.addEventListener("change", (event) => {
  enforceExclusive(event.target);
  updateCommand();
});
elements.launcher.addEventListener("change", updateCommand);
elements.copy.addEventListener("click", copyCommand);

elements.wrap.addEventListener("click", () => {
  state.wrapped = !state.wrapped;
  elements.output.classList.toggle("wrapped", state.wrapped);
  elements.wrap.setAttribute("aria-pressed", String(state.wrapped));
});

elements.reset.addEventListener("click", () => {
  elements.launcher.value = "venv";
  renderForm();
});

document.querySelector("#action-count").textContent = String(actions.length);
document.querySelector("#task-count").textContent = String(TASKS.length);
renderCategories();
renderActions();
renderForm();
