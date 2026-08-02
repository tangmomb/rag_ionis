# Arbre détaillé du pipeline vidéo

Ouvrir `index.html` directement dans un navigateur. La page reste locale et ne
nécessite aucune dépendance.

## Ce que montre l’interface

Cette page est un **simulateur documentaire** : elle n’analyse pas une vidéo.
Les boutons Oui / Non permettent de parcourir les routes réellement construites
par `pipeline/planner.py`.

Chaque carte détaille :

- la tâche ou la règle de routage ;
- les fichiers lus ;
- le traitement appliqué ;
- les fichiers créés ou mis à jour ;
- les règles essentielles du traitement ;
- le contenu exact envoyé au LLM lorsqu’un appel a lieu.

Le bouton **Masquer les détails** réduit l’arbre à ses titres et résumés. Sur
mobile, seule la branche active est affichée afin d’éviter un large défilement
horizontal.

## Entrées et source de vérité

Le pipeline part d’un dossier déjà ingéré qui contient au minimum :

- `VIDEO_ID.mp4`, jamais modifié ;
- `metadata/youtube_video_metadata.json`, qui fournit notamment la durée de
  référence ;
- `metadata/video_manifest.json`, créé ou repris par l’orchestrateur.

La durée de routage vient exclusivement des métadonnées YouTube. Le manifeste
est la source de vérité : il stocke les faits de routage, le plan, les statuts,
les artefacts et l’historique. Il est checkpointé atomiquement avant et après
chaque tâche.

## Routes représentées

| Condition | Traitement principal | Sortie de chunks |
|---|---|---|
| durée `> 600 s` | WhisperX sans diarisation, plain transcript, validation légère des speakers | `detail → section → global` |
| durée `≤ 600 s`, sous-titres détectés | inspection visuelle, OCR, WhisperX diarizé, rapprochement WhisperX/OCR | `detail` |
| durée `≤ 600 s`, sans sous-titres | inspection visuelle, OCR, WhisperX diarizé, correction déterministe par lexique OCR | `detail` |

Le seuil est strict : une vidéo de 600 secondes exactement reste courte.

Pour les vidéos courtes, le type visuel est déterminé dans cet ordre :

1. `interview` si le cluster DINO dominant couvre au moins 50 % des frames
   `footage` avec une similarité minimale de 0,88 ;
2. sinon `motion_design` si la durée est `< 180 s` et la part de `footage`
   `< 15 %` ;
3. sinon `video_recording`.

## Rapprochement WhisperX / OCR

Le point de convergence est `transcript.reconcile_ocr`.

Entrées exactes :

- `outputs/transcripts_whisper/transcript_1_brut.txt` ;
- `outputs/transcripts_ocr/plain_transcript.txt`.

Python extrait tous les corps de segments WhisperX sous la forme
`{index, text}`. Dans **une même requête**, le LLM reçoit cette liste complète
et le plain transcript OCR complet. Il peut corriger les noms, marques, mots mal
entendus ou manquants, accords et pluriels.

Les timecodes et labels `SPEAKER_XX` ne sont pas envoyés au LLM. Python :

1. exige chaque index une seule fois ;
2. refuse les index inconnus, manquants ou dupliqués ;
3. interdit qu’un segment source non vide revienne vide ;
4. réattache le timecode et le speaker d’origine.

Le nombre, l’ordre et les index des segments, les timecodes et les speakers
restent donc inchangés. Le modèle reçoit aussi l’instruction de ne pas
paraphraser, résumer ou ajouter un texte OCR non prononcé ; cette dernière
limite est une consigne sémantique, pas une vérification Python.

Sorties :

- `outputs/transcripts_whisper/transcript_2_corrected.txt` ;
- `outputs/transcripts_whisper/transcript_2_corrections.tsv`.

Si le plain transcript OCR n’existe pas, aucun LLM n’est appelé :
`transcript_2_corrected.txt` devient une copie de WhisperX et le TSV reste vide.

### Limite reflétée par l’interface

Le code vérifie la présence du fichier OCR, mais pas explicitement que son
contenu final est non vide. La question dans l’arbre porte donc sur un transcript
OCR **produit**, pas sur une qualité « exploitable » plus forte.

## Artefacts canoniques

Pour une vidéo courte :

| Fichier | Rôle |
|---|---|
| `transcript_1_brut.txt` | sortie WhisperX immuable |
| `transcript_2_corrected.txt` | WhisperX corrigé |
| `transcript_3_enriched.txt` | transcript timecodé avec noms de speakers et intercalaires `graphic` |
| `transcript_plain.txt` | texte sans timecodes, speakers ni intercalaires |
| `transcript_chunks.json` | chunks `detail` |

Pour une vidéo longue, seuls `transcript_1_brut.txt` et
`transcript_plain.txt` sont produits dans la chaîne canonique. Le JSON de chunks
contient ensuite les niveaux `detail`, `section` et `global`.

Le plain transcript OCR sert uniquement à la correction. Il n’est ni publié
directement, ni consommé par le RAG. Cas limite réel : si aucun texte parlé ne
reste sur une route courte, `transcript_plain.txt` peut être construit depuis
les catégories OCR `graphic` et `others`, avec un préfixe explicite. Les
sous-titres OCR restent exclus de ce repli.

## Après `pipeline run`

Le plan principal se termine aux chunks. Les embeddings sont créés par la tâche
explicite `embeddings.create`, puis la publication S3 et la synchronisation
PostgreSQL utilisent leurs commandes séparées.

L’arbre a été vérifié contre :

- `pipeline/planner.py` et `pipeline/step_handlers.py` ;
- les modules de `pipeline/steps/inspection/`, `ocr/`, `transcripts/`,
  `speakers/`, `chunks/` et `embeddings/` ;
- les tests ciblés de routage, rapprochement OCR, diarisation, speakers,
  plain transcript et chunking.
