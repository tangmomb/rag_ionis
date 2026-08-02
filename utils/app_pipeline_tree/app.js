"use strict";

const defaults = {
  longVideo: false,
  interview: false,
  motionDesign: false,
  subtitles: false,
};

const state = { ...defaults };
let detailsVisible = true;
let tooltipCounter = 0;

function artifact(path, description = "") {
  return { path, description };
}

function renderArtifacts(items = []) {
  if (!items.length) return "";
  return `
    <ul class="artifact-list">
      ${items
        .map(
          (item) => `
            <li>
              <code>${item.path}</code>
              ${item.description ? `<span>${item.description}</span>` : ""}
            </li>
          `,
        )
        .join("")}
    </ul>
  `;
}

function detailSection(title, content, className = "") {
  if (!content) return "";
  return `
    <section class="detail-section ${className}">
      <h4>${title}</h4>
      <div>${content}</div>
    </section>
  `;
}

function box(title, options = {}) {
  const classes = [
    "node-box",
    options.kind || "",
    options.phase ? `phase-${options.phase}` : "",
    options.llm ? "has-llm" : "",
    options.active === false ? "inactive" : "active",
  ]
    .filter(Boolean)
    .join(" ");

  const meta = [
    options.phase
      ? `<span class="phase-badge">${options.phaseLabel || options.phase}</span>`
      : "",
    options.task ? `<code class="task-id">${options.task}</code>` : "",
  ]
    .filter(Boolean)
    .join("");

  const llmBadge = options.llm
    ? `<span class="llm-badge" title="Appel à un LLM">✦ LLM</span>`
    : "";

  const tooltipId = options.tooltip
    ? options.tooltipId || `node-info-tooltip-${++tooltipCounter}`
    : "";
  const infoTooltip = options.tooltip
    ? `
      <span class="info-tooltip">
        <button
          type="button"
          class="info-tooltip-trigger"
          aria-label="${options.tooltipLabel || "Afficher l’explication"}"
          aria-describedby="${tooltipId}"
        >?</button>
        <span id="${tooltipId}" class="info-tooltip-content" role="tooltip">
          ${options.tooltip}
        </span>
      </span>
    `
    : "";

  const buttons = options.decisionKey
    ? `
      <div class="decision-buttons" aria-label="Réponse à la décision">
        <button
          type="button"
          data-decision="${options.decisionKey}"
          data-value="true"
          class="${state[options.decisionKey] ? "selected" : ""}"
          aria-pressed="${state[options.decisionKey]}"
        >Oui</button>
        <button
          type="button"
          data-decision="${options.decisionKey}"
          data-value="false"
          class="${!state[options.decisionKey] ? "selected" : ""}"
          aria-pressed="${!state[options.decisionKey]}"
        >Non</button>
      </div>
    `
    : "";

  const details = [
    detailSection(
      options.kind === "decision" ? "Règle" : "Action",
      options.action ? `<p>${options.action}</p>` : "",
      "process-section",
    ),
    options.llmCall
      ? `
        <aside class="llm-callout">
          <strong>Envoi au LLM</strong>
          <p>${options.llmCall}</p>
        </aside>
      `
      : "",
    detailSection(
      "Sorties",
      renderArtifacts(options.outputs),
      "outputs-section",
    ),
  ].join("");

  return `
    <article class="${classes}">
      <div class="node-meta">
        <div>${meta}</div>
        <div class="node-meta-actions">
          ${llmBadge}
          ${infoTooltip}
        </div>
      </div>
      <h3>${title}</h3>
      ${options.summary ? `<p class="node-summary">${options.summary}</p>` : ""}
      <div class="node-details">${details}</div>
      ${buttons}
    </article>
  `;
}

function oneChild(parent, child) {
  return `
    <div class="tree-node">
      ${parent}
      <div class="single-child">${child}</div>
    </div>
  `;
}

function branch(label, selected, content) {
  return `
    <div class="branch ${selected ? "" : "inactive"}">
      <span class="branch-label ${selected ? "selected" : ""}">${label}</span>
      ${content}
    </div>
  `;
}

function inactiveResult(text) {
  return `
    <div class="tree-node">
      ${box(text, {
        kind: "result",
        active: false,
        summary: "Branche non exécutée.",
      })}
    </div>
  `;
}

function decisionNode(
  question,
  decisionKey,
  yesContent,
  noContent,
  yesInactiveLabel,
  noInactiveLabel,
  options = {},
) {
  const yesSelected = state[decisionKey];
  const yesBranch = branch(
    "Oui",
    yesSelected,
    yesSelected ? yesContent : inactiveResult(yesInactiveLabel),
  );
  const noBranch = branch(
    "Non",
    !yesSelected,
    !yesSelected ? noContent : inactiveResult(noInactiveLabel),
  );
  return `
    <div class="tree-node">
      ${box(question, {
        ...options,
        kind: "decision",
        decisionKey,
      })}
      <div class="children">
        ${yesSelected ? `${yesBranch}${noBranch}` : `${noBranch}${yesBranch}`}
      </div>
    </div>
  `;
}

function shortChunkPipeline() {
  return `
    <div class="tree-node">
      ${box("Créer les chunks détail", {
        task: "chunks.create",
        phase: "structure",
        phaseLabel: "Données finales",
        kind: "result",
        summary: "Découpe le transcript en passages indexables.",
        inputs: [
          artifact("outputs/transcripts_whisper/transcript_plain.txt"),
        ],
        action:
          "Découpage par phrases, autour de 1 000 caractères. Aucun résumé sur la route courte.",
        outputs: [
          artifact("outputs/chunks/transcript_chunks.json", "niveau detail"),
        ],
      })}
    </div>
  `;
}

function shortSpeakerPipeline() {
  return oneChild(
    box("Proposer les speakers", {
      task: "speakers.propose",
      phase: "structure",
      phaseLabel: "Speakers",
      summary: "Prépare les noms candidats sans appeler de LLM.",
      inputs: [
        artifact("outputs/transcripts_whisper/transcript_2_corrected.txt"),
        artifact("outputs/ocr/01_processed_ocr_items.json"),
        artifact("outputs/ocr/02_filtered_ocr_overlays.json"),
      ],
      action:
        "Repère les noms dans les cartouches bas d’écran et les auto-présentations (« je m’appelle », « je suis », « moi c’est »).",
      outputs: [artifact("outputs/speakers/speaker_candidates.json")],
    }),
    oneChild(
      box("Valider les speakers", {
        task: "speakers.validate",
        phase: "structure",
        phaseLabel: "Speakers",
        llm: true,
        summary: "Garde les personnes physiques et cherche leur fonction.",
        inputs: [artifact("outputs/speakers/speaker_candidates.json")],
        action:
          "Le LLM vérifie les candidats et retourne des couples nom / fonction en JSON.",
        llmCall:
          "Textes OCR hors sous-titres et noms candidats. Sur cette route courte, l’extrait de transcript envoyé est vide.",
        tooltipLabel: "Voir un exemple du prompt de validation des speakers",
        tooltip:
          "Le prompt demande de ne conserver que les personnes physiques et de retourner leur poste avec leur entreprise lorsque celle-ci est identifiable. <span class=\"info-tooltip-example\"><strong>Exemple simplifié du prompt</strong><code><b>SYSTEM</b> — Vérifie les speakers détectés. Rejette entreprises, écoles, métiers, lieux, slogans et OCR parasites. Retourne uniquement un JSON <b>{speakers: [{speaker, title}]}</b>.<br><br><b>USER</b> — Identifie les personnes physiques et leur fonction.<br>{<br>&nbsp;&nbsp;&quot;transcript_excerpt&quot;: &quot;&quot;,<br>&nbsp;&nbsp;&quot;ocr_detected_texts&quot;: [&quot;Sophie Martin — Directrice IA, Ionis-STM&quot;],<br>&nbsp;&nbsp;&quot;candidates&quot;: [{&quot;name&quot;: &quot;Sophie Martin&quot;}]<br>}</code></span>",
        outputs: [artifact("outputs/speakers/speakers_validated.json")],
      }),
      oneChild(
        box("Enrichir le transcript", {
          task: "transcript.enrich",
          phase: "structure",
          phaseLabel: "Finalisation",
          summary: "Remplace SPEAKER_XX par les noms validés.",
          inputs: [
            artifact(
              "outputs/transcripts_whisper/transcript_2_corrected.txt",
            ),
            artifact("outputs/speakers/speakers_validated.json"),
            artifact("outputs/ocr/02_filtered_ocr_overlays.json"),
          ],
          action:
            "Associe les voix aux personnes et ajoute uniquement les textes OCR de type graphic comme INTERCALAIRE.",
          outputs: [
            artifact(
              "outputs/transcripts_whisper/transcript_3_enriched.txt",
            ),
          ],
        }),
        oneChild(
          box("Créer le transcript sans timecodes", {
            task: "transcript.create_plain",
            phase: "structure",
            phaseLabel: "Finalisation",
            summary: "Produit le texte utilisé par le RAG.",
            inputs: [
              artifact(
                "outputs/transcripts_whisper/transcript_3_enriched.txt",
              ),
            ],
            action:
              "Retire timecodes, speakers et intercalaires. Si aucune parole ne reste, utilise graphic + others comme repli visuel.",
            outputs: [
              artifact(
                "outputs/transcripts_whisper/transcript_plain.txt",
              ),
            ],
          }),
          shortChunkPipeline(),
        ),
      ),
    ),
  );
}

function reconcileWithOcrPipeline() {
  return oneChild(
    box("Normaliser Ionis-STM", {
      task: "transcript.normalize_brand",
      phase: "transcript",
      phaseLabel: "Transcript OCR",
      summary: "Corrige les variantes connues de la marque, sans LLM.",
      inputs: [artifact("outputs/ocr/ocr_subtitles_timecoded.txt")],
      action: "Applique des remplacements déterministes dans le fichier OCR.",
      outputs: [artifact("outputs/ocr/ocr_subtitles_timecoded.txt")],
    }),
    oneChild(
      box("Créer le transcript OCR plain", {
        task: "transcript.create_plain_ocr",
        phase: "transcript",
        phaseLabel: "Transcript OCR",
        summary: "Prépare la référence OCR envoyée au LLM.",
        inputs: [artifact("outputs/ocr/ocr_subtitles_timecoded.txt")],
        action: "Retire les timecodes et concatène les sous-titres.",
        outputs: [
          artifact("outputs/transcripts_ocr/plain_transcript.txt"),
        ],
      }),
      oneChild(
        box("Rapprocher WhisperX et OCR", {
          task: "transcript.reconcile_ocr",
          phase: "transcript",
          phaseLabel: "Correction",
          llm: true,
          summary:
            "Le LLM compare les deux transcripts et corrige le texte WhisperX.",
          inputs: [
            artifact(
              "outputs/transcripts_whisper/transcript_1_brut.txt",
              "segments WhisperX",
            ),
            artifact(
              "outputs/transcripts_ocr/plain_transcript.txt",
              "transcript OCR complet",
            ),
          ],
          action:
            "Python lit <code>outputs/transcripts_whisper/transcript_1_brut.txt</code>, en extrait chaque corps de segment sous la forme {index, text} et retire donc les timecodes et labels SPEAKER_XX de la requête. Il lit aussi intégralement <code>outputs/transcripts_ocr/plain_transcript.txt</code>. Ces deux contenus sont réunis dans une seule requête au LLM. Python réattache ensuite à chaque texte corrigé les timecodes et speakers d’origine.",
          llmCall:
            "Envoi conjoint des segments extraits de <code>transcript_1_brut.txt</code> et du contenu complet de <code>plain_transcript.txt</code>. Le LLM renvoie uniquement {index, text} pour chaque segment WhisperX.",
          tooltipId: "reconcile-segment-reassembly",
          tooltipLabel: "Comment Python réattache les timecodes et speakers",
          tooltip:
            "Python mémorise, pour chaque index envoyé, la ligne d’origine, son préfixe de timecode et son éventuel label <code>SPEAKER_XX</code>. Après avoir vérifié que chaque index revient exactement une fois, il remplace uniquement le corps du segment : <code>préfixe original + speaker original + texte corrigé</code>. Les lignes non reconnues comme segments WhisperX restent inchangées. <span class=\"info-tooltip-example\"><strong>Exemple simplifié du prompt</strong><code><b>SYSTEM</b> — Corrige la transcription WhisperX à partir des sous-titres OCR. Conserve tous les segments, leurs index et leur ordre.<br><br><b>USER</b> — SEGMENTS WHISPERX : [{&quot;index&quot;: 0, &quot;text&quot;: &quot;Bienvenue à Ionis STM.&quot;}]<br>TRANSCRIPT OCR : Bienvenue à Ionis-STM.</code></span>",
          outputs: [
            artifact(
              "outputs/transcripts_whisper/transcript_2_corrected.txt",
            ),
            artifact(
              "outputs/transcripts_whisper/transcript_2_corrections.tsv",
            ),
          ],
        }),
        shortSpeakerPipeline(),
      ),
    ),
  );
}

function ocrTranscriptPipeline() {
  return oneChild(
    box("Construire le transcript OCR timecodé", {
      task: "transcript.extract_ocr",
      phase: "transcript",
      phaseLabel: "Transcript OCR",
      summary: "Construit la référence depuis les sous-titres détectés.",
      inputs: [artifact("outputs/ocr/01_processed_ocr_items.json")],
      action:
        "Garde les éléments kind=subtitle, les trie et retire les doublons.",
      outputs: [
        artifact("outputs/ocr/ocr_subtitles_timecoded.txt"),
      ],
    }),
    reconcileWithOcrPipeline(),
  );
}

function noSubtitleCorrectionPipeline() {
  return oneChild(
    box("Corriger WhisperX avec le lexique OCR", {
      task: "transcript.correct_whisper",
      phase: "transcript",
      phaseLabel: "Correction",
      summary: "Corrige surtout les noms propres, sans LLM.",
      inputs: [
        artifact("outputs/transcripts_whisper/transcript_1_brut.txt"),
        artifact("outputs/ocr/01_processed_ocr_items.json"),
      ],
      action:
        "Construit un lexique avec les textes visuels et applique les correspondances fiables.",
      outputs: [
        artifact(
          "outputs/transcripts_whisper/transcript_2_corrected.txt",
        ),
        artifact(
          "outputs/transcripts_whisper/transcript_2_corrections.tsv",
        ),
      ],
    }),
    shortSpeakerPipeline(),
  );
}

function commonShortProcessing(nextStep, hasSubtitles) {
  const subtitleClassification = hasSubtitles
    ? "La détection précédente a validé une zone stable : les OCR proches de cette ancre sont classés subtitle."
    : "La détection précédente n’a validé aucune zone stable : les détections initialement susceptibles d’être des sous-titres sont reclassées others et aucun transcript OCR de sous-titres n’est construit sur cette route.";

  return oneChild(
    box("Construire l’OCR traité", {
      task: "ocr.build_processed",
      phase: "inspection",
      phaseLabel: "OCR",
      summary: hasSubtitles
        ? "Nettoie et classe les textes, dont les sous-titres détectés."
        : "Nettoie les textes sans retenir de zone de sous-titres.",
      inputs: [
        artifact("outputs/ocr/raw/raw_ocr_footage_frames.json"),
        artifact("outputs/ocr/raw/raw_ocr_graphic_frames.json"),
        artifact("outputs/ocr/raw/raw_ocr_mixture_frames.json"),
      ],
      action:
        `Garde les scores ≥ 0,90. ${subtitleClassification} Les OCR issus du dossier images/graphic sont classés graphic ; les autres sont classés others. <strong>Décors statiques :</strong> hors subtitles et graphics, les textes identiques ou proches sont regroupés lorsque le centre de leur box reste à ± 4,5 % sur les axes x et y. Un groupe présent sur plus de 20 images distinctes est considéré comme un décor et supprimé. Deux textes sont proches s’ils sont identiques ou, à partir de 8 caractères, si l’un contient au moins 65 % de l’autre ou si leur similarité atteint 0,82. Les libellés décoratifs connus comme ionis, stm, x ou in sont aussi retirés. <strong>Doublons :</strong> après normalisation de la casse, des espaces et de la ponctuation extérieure, deux items ayant le même texte à la même seconde sont fusionnés ; seul le premier dans l’ordre des frames et des boxes est conservé.`,
      outputs: [
        artifact("outputs/ocr/01_processed_ocr_items.json"),
      ],
    }),
    oneChild(
      box("Filtrer les overlays OCR", {
        task: "ocr.filter_overlays",
        phase: "inspection",
        phaseLabel: "OCR",
        summary: "Regroupe les textes visuels utiles.",
        inputs: [artifact("outputs/ocr/01_processed_ocr_items.json")],
        action:
          "Fusionne les fragments proches et organise les textes par catégorie et timecode.",
        outputs: [
          artifact("outputs/ocr/02_filtered_ocr_overlays.json"),
        ],
      }),
      oneChild(
        box("Transcrire avec WhisperX", {
          task: "transcript.whisper",
          phase: "transcript",
          phaseLabel: "Audio",
          summary: "Crée le transcript brut de référence.",
          inputs: [artifact("VIDEO_ID.mp4")],
          action:
            "Extrait l’audio et le transcrit avec WhisperX. Chaque mot est synchronisé avec son timecode. Pour les vidéos de 600 secondes ou moins, Pyannote attribue aussi un speaker aux paroles.",
          outputs: [
            artifact("outputs/transcripts_whisper/audio/VIDEO_ID.mp3"),
            artifact(
              "outputs/transcripts_whisper/transcript_1_brut.txt",
            ),
          ],
        }),
        nextStep,
      ),
    ),
  );
}

function subtitleDecision() {
  return decisionNode(
    "Des sous-titres incrustés sont-ils détectés ?",
    "subtitles",
    commonShortProcessing(ocrTranscriptPipeline(), true),
    commonShortProcessing(noSubtitleCorrectionPipeline(), false),
    "Route avec transcript OCR",
    "Route avec lexique OCR",
    {
      task: "video.detect_subtitles",
      phase: "routing",
      phaseLabel: "Décision",
      summary: "Recherche une zone de sous-titres stable.",
      inputs: [artifact("outputs/ocr/ocr_box_locations.json")],
      action:
        "Zone stable pendant au moins 10 s en continu, avec au moins 3 textes différents.",
    },
  );
}

function typeResult(type) {
  const labels = {
    interview: "Interview",
    motion_design: "Motion design",
    video_recording: "Captation vidéo",
  };

  return oneChild(
    box(`Type retenu : ${labels[type]}`, {
      task: "video.infer_type",
      phase: "routing",
      phaseLabel: "Type",
      kind: "result",
      summary: `Enregistre ${type} dans la route.`,
      inputs: [
        artifact("outputs/images/frame_classification_manifest.json"),
        artifact("outputs/interview/interview_detection_manifest.json"),
      ],
    }),
    oneChild(
      box("Extraire l’OCR brut", {
        task: "ocr.extract_raw",
        phase: "inspection",
        phaseLabel: "OCR",
        summary: "Lit les textes dans les trois familles de frames.",
        inputs: [
          artifact("outputs/images/footage/*.jpg"),
          artifact("outputs/images/graphic/*.jpg"),
          artifact("outputs/images/mixture/*.jpg"),
        ],
        action:
          "PaddleOCR extrait texte, position et score. Sous Windows, il tourne dans un processus isolé.",
        outputs: [
          artifact("outputs/ocr/raw/raw_ocr_footage_frames.json"),
          artifact("outputs/ocr/raw/raw_ocr_graphic_frames.json"),
          artifact("outputs/ocr/raw/raw_ocr_mixture_frames.json"),
        ],
      }),
      oneChild(
        box("Extraire les positions OCR", {
          task: "ocr.extract_boxes",
          phase: "inspection",
          phaseLabel: "OCR",
          summary: "Associe chaque texte à une position et un timecode.",
          inputs: [artifact("outputs/ocr/raw/raw_ocr_*_frames.json")],
          action: "Normalise les boxes OCR et récupère la seconde de la frame.",
          outputs: [artifact("outputs/ocr/ocr_box_locations.json")],
        }),
        subtitleDecision(),
      ),
    ),
  );
}

function motionDesignDecision() {
  return decisionNode(
    "Durée < 180 s et footage < 15 % ?",
    "motionDesign",
    typeResult("motion_design"),
    typeResult("video_recording"),
    "Classer motion design",
    "Classer captation vidéo",
    {
      phase: "routing",
      phaseLabel: "Décision",
      summary: "Sépare motion design et captation.",
      inputs: [
        artifact("metadata/youtube_video_metadata.json"),
        artifact("outputs/images/frame_classification_manifest.json"),
      ],
      action:
        "Les deux seuils sont stricts. Sans frame footage, la règle motion design est satisfaite.",
    },
  );
}

function interviewDecision() {
  return decisionNode(
    "Une interview est-elle détectée ?",
    "interview",
    typeResult("interview"),
    motionDesignDecision(),
    "Classer interview",
    "Tester motion design",
    {
      task: "video.detect_interview",
      phase: "routing",
      phaseLabel: "Décision",
      summary: "Cherche un plan filmé dominant.",
      inputs: [
        artifact("outputs/images/footage/*.jpg"),
        artifact("outputs/images/frame_classification_features.json"),
      ],
      action:
        "Les embeddings DINO des frames footage sont normalisés en L2. Le produit scalaire calcule ensuite la similarité cosinus entre chaque paire. Pour chaque frame, on compte les voisines à ≥ 0,88 : celle qui en possède le plus devient le centre du cluster dominant. En cas d’égalité, la meilleure similarité moyenne l’emporte. Interview si ce cluster couvre au moins 50 % des frames footage.",
      outputs: [
        artifact(
          "outputs/interview/interview_detection_manifest.json",
        ),
      ],
    },
  );
}

function shortPipeline() {
  return oneChild(
    box("Extraire les frames", {
      task: "frames.extract",
      phase: "inspection",
      phaseLabel: "Images",
      summary: "Échantillonne la vidéo toutes les 0,5 seconde.",
      inputs: [artifact("VIDEO_ID.mp4")],
      action: "FFmpeg crée des JPEG nommés avec leur timecode.",
      outputs: [artifact("outputs/images/*.jpg")],
    }),
    oneChild(
      box("Classifier les frames", {
        task: "frames.classify",
        phase: "inspection",
        phaseLabel: "Images",
        summary: "Classe footage, graphic ou mixture.",
        inputs: [artifact("outputs/images/*.jpg")],
        action:
          "DINOv2 et CLIP produisent chacun un vecteur global par frame : 768 dimensions pour DINOv2 et 512 pour CLIP. Après normalisation L2, les deux vecteurs sont concaténés en un vecteur de 1 280 dimensions. Le modèle local chargé depuis le fichier Joblib attend exactement ces 1 280 features pour classer l’image en footage, graphic ou mixture.",
        outputs: [
          artifact(
            "outputs/images/frame_classification_manifest.json",
          ),
          artifact(
            "outputs/images/frame_classification_features.json",
          ),
          artifact("outputs/images/{footage,graphic,mixture}/*.jpg"),
        ],
      }),
      interviewDecision(),
    ),
  );
}

function longPipeline() {
  return oneChild(
    box("Type retenu : long_video", {
      task: "video.infer_type",
      phase: "routing",
      phaseLabel: "Type",
      kind: "result",
      summary: "Ignore toute l’inspection visuelle et l’OCR.",
      inputs: [artifact("metadata/youtube_video_metadata.json")],
      action: "Enregistre directement le type long_video.",
    }),
    oneChild(
      box("Transcrire avec WhisperX", {
        task: "transcript.whisper",
        phase: "transcript",
        phaseLabel: "Audio",
        summary: "Transcrit et aligne l’audio sans diarisation.",
        inputs: [artifact("VIDEO_ID.mp4")],
        action: "Extrait le MP3 puis produit le transcript brut timecodé.",
        outputs: [
          artifact("outputs/transcripts_whisper/audio/VIDEO_ID.mp3"),
          artifact(
            "outputs/transcripts_whisper/transcript_1_brut.txt",
          ),
        ],
      }),
      oneChild(
        box("Créer le transcript plain", {
          task: "transcript.create_plain",
          phase: "transcript",
          phaseLabel: "Transcript",
          summary: "Retire les timecodes du WhisperX brut.",
          inputs: [
            artifact(
              "outputs/transcripts_whisper/transcript_1_brut.txt",
            ),
          ],
          action: "Utilise toujours le brut sur la route longue.",
          outputs: [
            artifact(
              "outputs/transcripts_whisper/transcript_plain.txt",
            ),
          ],
        }),
        oneChild(
          box("Préparer les speakers", {
            task: "speakers.propose",
            phase: "structure",
            phaseLabel: "Speakers",
            summary: "Prépare un contexte léger, sans LLM.",
            inputs: [
              artifact(
                "outputs/transcripts_whisper/transcript_plain.txt",
              ),
            ],
            action: "Conserve le titre et les 1 000 premiers caractères.",
            outputs: [
              artifact("outputs/speakers/speaker_candidates.json"),
            ],
          }),
          oneChild(
            box("Valider les speakers", {
              task: "speakers.validate",
              phase: "structure",
              phaseLabel: "Speakers",
              llm: true,
              summary: "Cherche les personnes et leur fonction.",
              inputs: [
                artifact("outputs/speakers/speaker_candidates.json"),
              ],
              action: "Le LLM retourne une liste structurée nom / fonction.",
              llmCall:
                "Les 1 000 premiers caractères du transcript et les noms candidats. Aucun texte OCR n’est disponible sur la route longue.",
              tooltipLabel: "Voir un exemple du prompt de validation des speakers",
              tooltip:
                "Le même validateur recherche les personnes dans l’extrait du transcript, même si elles ne figurent pas encore parmi les candidats. <span class=\"info-tooltip-example\"><strong>Exemple simplifié du prompt</strong><code><b>SYSTEM</b> — Garde uniquement les personnes physiques. Pour chacune, retourne <b>speaker</b> et <b>title</b>, en incluant l’entreprise lorsqu’elle est identifiable.<br><br><b>USER</b> — Identifie les speakers dans ces sources.<br>{<br>&nbsp;&nbsp;&quot;transcript_excerpt&quot;: &quot;Bonjour, je suis Marc Dupont, responsable innovation chez Example Corp…&quot;,<br>&nbsp;&nbsp;&quot;ocr_detected_texts&quot;: [],<br>&nbsp;&nbsp;&quot;candidates&quot;: [{&quot;name&quot;: &quot;Marc Dupont&quot;}]<br>}</code></span>",
              outputs: [
                artifact("outputs/speakers/speakers_validated.json"),
              ],
            }),
            oneChild(
              box("Créer les chunks détail", {
                task: "chunks.create",
                phase: "structure",
                phaseLabel: "Chunks",
                summary: "Découpe le transcript par phrases.",
                inputs: [
                  artifact(
                    "outputs/transcripts_whisper/transcript_plain.txt",
                  ),
                ],
                action: "Crée des passages autour de 1 000 caractères.",
                outputs: [
                  artifact(
                    "outputs/chunks/transcript_chunks.json",
                    "niveau detail",
                  ),
                ],
              }),
              oneChild(
                box("Résumer les sections", {
                  task: "chunks.summarize_sections",
                  phase: "structure",
                  phaseLabel: "Chunks",
                  llm: true,
                  summary: "Résume chaque groupe de 6 chunks détail.",
                  inputs: [
                    artifact("outputs/chunks/transcript_chunks.json"),
                  ],
                  action: "Ajoute les chunks de niveau section.",
                  llmCall: "Le contenu de chaque groupe de 6 chunks.",
                  tooltipLabel: "Voir un exemple du prompt de résumé de section",
                  tooltip:
                    "Les contenus des 6 chunks détail sont concaténés avec une ligne vide entre eux. Une requête indépendante est construite pour chaque groupe. <span class=\"info-tooltip-example\"><strong>Exemple simplifié du prompt</strong><code><b>SYSTEM</b> — Résume un transcript en français. Produis un résumé fidèle, clair et autonome. N’invente aucune information. Retourne uniquement <b>{summary}</b>.<br><br><b>USER</b> — Résume le texte en 4 phrases maximum et 1 200 caractères maximum.<br><br>TEXTE :<br>[contenu du chunk détail 1]<br><br>…<br><br>[contenu du chunk détail 6]</code></span>",
                  outputs: [
                    artifact(
                      "outputs/chunks/transcript_chunks.json",
                      "detail + section",
                    ),
                  ],
                }),
                `
                  <div class="tree-node">
                    ${box("Créer le résumé global", {
                      task: "chunks.summarize_video",
                      phase: "structure",
                      phaseLabel: "Données finales",
                      kind: "result",
                      llm: true,
                      summary: "Résume les sections en un chunk global.",
                      inputs: [
                        artifact("outputs/chunks/transcript_chunks.json"),
                      ],
                      action: "Ajoute la hiérarchie global → section → detail.",
                      llmCall: "Les résumés de section.",
                      tooltipLabel: "Voir un exemple du prompt de résumé global",
                      tooltip:
                        "Tous les résumés de section sont concaténés dans leur ordre, avec une ligne vide entre eux, puis envoyés dans une seule requête. <span class=\"info-tooltip-example\"><strong>Exemple simplifié du prompt</strong><code><b>SYSTEM</b> — Résume un transcript en français. Produis un résumé fidèle, clair et autonome. N’invente aucune information. Retourne uniquement <b>{summary}</b>.<br><br><b>USER</b> — Résume le texte en 6 phrases maximum et 1 800 caractères maximum.<br><br>TEXTE :<br>[résumé de la section 1]<br><br>…<br><br>[résumé de la dernière section]</code></span>",
                      outputs: [
                        artifact("outputs/chunks/transcript_chunks.json"),
                      ],
                    })}
                  </div>
                `,
              ),
            ),
          ),
        ),
      ),
    ),
  );
}

function durationDecision() {
  return decisionNode(
    "La vidéo dépasse-t-elle 600 secondes ?",
    "longVideo",
    longPipeline(),
    shortPipeline(),
    "Route longue",
    "Route courte",
    {
      phase: "routing",
      phaseLabel: "Décision",
      summary: "Choisit la route courte ou longue.",
      inputs: [artifact("metadata/youtube_video_metadata.json")],
      action:
        "Lit duration_seconds. Le seuil est strict : 600 s reste une vidéo courte.",
    },
  );
}

function renderTree() {
  const start = oneChild(
    box("Initialiser la vidéo", {
      task: "PipelineContext.inspect",
      phase: "routing",
      phaseLabel: "Initialisation",
      kind: "start",
      summary: "Charge la vidéo, ses métadonnées et son manifeste.",
      inputs: [
        artifact("VIDEO_ID.mp4"),
        artifact("metadata/youtube_video_metadata.json"),
      ],
      action:
        "Sonde le fichier, construit le plan et met à jour le manifeste après chaque tâche.",
    }),
    durationDecision(),
  );
  document.querySelector("#pipeline-tree").innerHTML = start;
}

function render() {
  renderTree();
}

function setDetailsVisibility(visible) {
  detailsVisible = visible;
  document.body.classList.toggle("details-hidden", !visible);
  const button = document.querySelector("#details-button");
  button.setAttribute("aria-pressed", String(visible));
  button.textContent = visible ? "Masquer les détails" : "Afficher les détails";
}

document.querySelector("#pipeline-tree").addEventListener("click", (event) => {
  const button = event.target.closest("[data-decision]");
  if (!button) return;
  const key = button.dataset.decision;
  const value = button.dataset.value;
  state[key] = value === "true";
  render();
  document
    .querySelector(`[data-decision="${key}"][data-value="${value}"]`)
    ?.focus();
});

document.querySelector("#details-button").addEventListener("click", () => {
  setDetailsVisibility(!detailsVisible);
});

document.querySelector("#reset-button").addEventListener("click", () => {
  Object.assign(state, defaults);
  render();
});

render();
setDetailsVisibility(true);
