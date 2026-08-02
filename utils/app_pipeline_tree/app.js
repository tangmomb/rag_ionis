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

  const tooltipContent = options.tooltip || options.technical;
  const tooltipId = tooltipContent
    ? options.tooltipId || `node-info-tooltip-${++tooltipCounter}`
    : "";
  const infoTooltip = tooltipContent
    ? `
      <span class="info-tooltip">
        <button
          type="button"
          class="info-tooltip-trigger"
          aria-label="${options.tooltipLabel || "Afficher l’explication"}"
          aria-describedby="${tooltipId}"
        >?</button>
        <span id="${tooltipId}" class="info-tooltip-content" role="tooltip">
          ${tooltipContent}
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
      "Entrées",
      renderArtifacts(options.inputs),
      "inputs-section",
    ),
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
        action: "Crée les chunks de niveau detail.",
        technical:
          "Découpe le transcript par phrases en passages d’environ 1 000 caractères. La route courte ne crée aucun résumé de niveau section ou global.",
        outputs: [
          artifact("outputs/chunks/transcript_chunks.json", "niveau detail"),
        ],
      })}
    </div>
  `;
}

function shortSpeakerPipeline() {
  return oneChild(
    box("Extraire les candidats speakers", {
      task: "speakers.propose",
      phase: "structure",
      phaseLabel: "Speakers",
      summary: "Repère localement les noms possibles, sans appeler de LLM.",
      inputs: [
        artifact("outputs/transcripts_whisper/transcript_2_corrected.txt"),
        artifact("outputs/ocr/01_processed_ocr_items.json"),
        artifact("outputs/ocr/02_filtered_ocr_overlays.json"),
      ],
      action: "Extrait les candidats depuis le transcript et l’OCR.",
      technical:
        "Analyse tout <code>transcript_2_corrected.txt</code> pour extraire les noms qui suivent « je m’appelle », « je suis » ou « moi c’est », puis ajoute les noms trouvés dans les cartouches OCR bas d’écran. Cette étape écrit les candidats dans <code>speaker_candidates.json</code> sans rien envoyer au LLM. Le titre, le transcript complet et les textes OCR y sont seulement stockés pour l’étape de validation suivante.",
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
        action: "Envoie le contexte au LLM et valide les speakers.",
        llmCall:
          "Titre de la vidéo, transcript WhisperX corrigé complet, textes OCR hors sous-titres et candidats extraits des auto-présentations ou des cartouches OCR.",
        tooltipLabel: "Voir le prompt de validation des speakers",
        tooltip:
          "<span class=\"info-tooltip-example prompt-only\"><strong>Exemple simplifié du prompt</strong><code><b>SYSTEM</b> — Vérifie les speakers détectés à partir du titre, du transcript, de l’OCR et des candidats. Retourne uniquement un JSON <b>{speakers: [{speaker, title}]}</b>.<br><br><b>USER</b> — Identifie les personnes physiques et leur fonction.<br>{<br>&nbsp;&nbsp;&quot;video_title&quot;: &quot;Sophie Martin présente son métier&quot;,<br>&nbsp;&nbsp;&quot;transcript_excerpt&quot;: &quot;[00:02-00:06] SPEAKER_00: Bonjour, je suis Sophie Martin, directrice IA…&quot;,<br>&nbsp;&nbsp;&quot;ocr_detected_texts&quot;: [&quot;Sophie Martin — Directrice IA, Ionis-STM&quot;],<br>&nbsp;&nbsp;&quot;candidates&quot;: [{&quot;name&quot;: &quot;Sophie Martin&quot;}]<br>}</code></span>",
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
          action: "Associe les noms aux voix et ajoute les intercalaires.",
          technical:
            "Pour chaque nom validé retrouvé dans un OCR de type <code>name</code>, <code>lower_third</code> ou <code>others</code>, Python utilise le timecode de son apparition à l’écran et l’associe au segment <code>SPEAKER_XX</code> qui parle à cet instant. Si aucun segment ne contient exactement ce timecode, le segment le plus proche est accepté jusqu’à 2 secondes. Les correspondances nom–label sont cumulées, classées par score et rendues uniques avant de remplacer les labels <code>SPEAKER_XX</code> dans le transcript. Les auto-présentations comme « je m’appelle » complètent ces indices. Enfin, seuls les textes OCR de type <code>graphic</code> sont ajoutés sous forme d’<code>INTERCALAIRE</code> à leur propre timecode. Les suffixes de doublon comme <code>00:03#2</code> sont interprétés comme le même instant <code>00:03</code>, ce qui permet de conserver plusieurs intercalaires simultanés sans erreur.",
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
            action: "Produit le transcript plain utilisé par le RAG.",
            technical:
              "Retire les timecodes, les noms de speakers et les intercalaires. Si aucune parole ne reste, construit un repli visuel depuis les catégories OCR <code>graphic</code> et <code>others</code>.",
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
      action: "Normalise les variantes de la marque.",
      technical:
        "Utilise la règle partagée <code>pipeline/support/brand_normalization.py</code>, commune aux branches OCR et Whisper. Elle remplace notamment « Ionis STM », « Onis-STM », « L’Onis STM », « Lonis STM », « Yonis STM », « Yaunis STM », « Unisystem », « UNISSTM » et « IonisSTM » par <code>Ionis-STM</code>, puis réécrit le même fichier.",
      outputs: [artifact("outputs/ocr/ocr_subtitles_timecoded.txt")],
    }),
    oneChild(
      box("Créer le transcript OCR plain", {
        task: "transcript.create_plain_ocr",
        phase: "transcript",
        phaseLabel: "Transcript OCR",
      summary: "Prépare la référence OCR envoyée au LLM.",
      inputs: [artifact("outputs/ocr/ocr_subtitles_timecoded.txt")],
      action: "Produit le transcript OCR sans timecodes.",
      technical:
        "Retire les timecodes de <code>ocr_subtitles_timecoded.txt</code>, concatène les sous-titres et écrit le résultat dans <code>outputs/transcripts_ocr/plain_transcript.txt</code>.",
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
          action: "Corrige WhisperX à partir du transcript OCR.",
          llmCall:
            "Envoi conjoint des segments extraits de <code>transcript_1_brut.txt</code> et du contenu complet de <code>plain_transcript.txt</code>. Le LLM renvoie uniquement {index, text} pour chaque segment WhisperX.",
          tooltipId: "reconcile-segment-reassembly",
          tooltipLabel: "Comment Python réattache les timecodes et speakers",
          tooltip:
            "Python lit <code>transcript_1_brut.txt</code>, en extrait les corps de segments sous la forme {index, text} sans envoyer les timecodes ni les labels <code>SPEAKER_XX</code>, puis ajoute le contenu complet de <code>plain_transcript.txt</code> OCR à la même requête. Pour chaque index, il mémorise la ligne d’origine, son préfixe de timecode et son éventuel speaker. Après validation de tous les index retournés, il reconstruit chaque ligne avec <code>préfixe original + speaker original + texte corrigé</code>. Les lignes non reconnues comme segments WhisperX restent inchangées. <span class=\"info-tooltip-example\"><strong>Exemple simplifié du prompt</strong><code><b>SYSTEM</b> — Corrige la transcription WhisperX à partir des sous-titres OCR. Conserve tous les segments, leurs index et leur ordre.<br><br><b>USER</b> — SEGMENTS WHISPERX : [{&quot;index&quot;: 0, &quot;text&quot;: &quot;Bienvenue à Ionis STM.&quot;}]<br>TRANSCRIPT OCR : Bienvenue à Ionis-STM.</code></span>",
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
      action: "Produit le transcript OCR timecodé.",
      technical:
        "Sélectionne les éléments <code>kind=subtitle</code> de l’OCR traité, les trie chronologiquement et retire les doublons avant d’écrire <code>ocr_subtitles_timecoded.txt</code>.",
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
      action: "Corrige WhisperX avec le lexique visuel.",
      technical:
        "Applique d’abord la normalisation Ionis-STM partagée avec la branche OCR, puis construit un lexique déterministe depuis les textes OCR de type <code>name</code>, <code>lower_third</code>, <code>title</code>, <code>logo</code>, <code>graphic</code> ou <code>outro</code>. En mode balanced, une forme doit apparaître au moins 2 fois. Les noms propres proches sont ensuite corrigés selon leur similarité ; les mots de 4 caractères ou moins demandent au moins 90 %. Le transcript corrigé et le rapport TSV sont écrits séparément.",
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
    : "La détection précédente n’a validé aucune zone stable : aucune catégorie subtitle n’est retenue pour cette route ; les textes sont classés graphic ou others et aucun transcript OCR de sous-titres n’est construit.";

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
      action: "Classe, nettoie et déduplique les textes OCR.",
      technical:
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
        action: "Regroupe les overlays OCR utiles.",
        technical:
          "<strong>Fragments d’un même timecode :</strong> les textes appartenant à la même famille d’overlay sont concaténés dans leur ordre OCR. Les fragments d’un visuel <code>graphic</code> sont séparés par un espace ; ceux des catégories <code>name</code>, <code>lower_third</code> et <code>title</code> sont séparés par <code> / </code>. Si plusieurs overlays distincts doivent rester au même instant, les clés suivantes reçoivent un suffixe <code>#2</code>, <code>#3</code>, etc. <strong>Lectures successives :</strong> pour un overlay sur footage vu sur des frames espacées d’au plus 1 seconde, si un texte répète ou prolonge l’autre, seule la lecture la plus complète est conservée. Un fragment court est également supprimé lorsqu’il est inclus dans une version plus longue détectée dans les 4 secondes suivantes. Pour les graphics séparés de moins de 5 secondes, les lectures identiques ou incluses sont regroupées ; à partir de 6 caractères, le regroupement accepte aussi jusqu’à 2 caractères différents ou une similarité d’au moins 0,62. La version la plus complète d’un groupe observé sur au moins 2 images est gardée. Enfin, les doublons exacts normalisés sont retirés. Le résultat est organisé par catégorie et par timecode dans <code>02_filtered_ocr_overlays.json</code>.",
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
          action: "Produit le transcript WhisperX brut diarizé.",
          technical:
            "Extrait l’audio en MP3, transcrit et aligne chaque mot avec WhisperX, puis utilise Pyannote sur les vidéos de 600 secondes ou moins pour attribuer un label <code>SPEAKER_XX</code> aux paroles.",
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
      action: "Décide si une zone de sous-titres est stable.",
      technical:
        "Ne considère que les boxes OCR dont le score est ≥ 0,90. Une série est validée si elle reste stable pendant au moins 10 secondes continues, contient au moins 3 variantes de texte et si son centre horizontal se situe entre 40 % et 60 % de la largeur vidéo, soit à ± 10 % du centre. La position verticale reste libre.",
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
        action: "Extrait le texte de chaque frame.",
        technical:
          "PaddleOCR extrait pour chaque détection le texte, la position et le score de confiance dans les trois familles de frames. Sous Windows, l’inférence s’exécute dans un processus isolé.",
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
          action: "Associe chaque OCR à sa position et son timecode.",
          technical:
            "Normalise les coordonnées des boxes OCR, déduit la seconde correspondante depuis le nom timecodé de chaque frame et conserve l’alignement exact entre chaque box, son texte et son score de confiance. Ces scores alimentent ensuite le seuil de 90 % de la détection des sous-titres.",
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
      action: "Choisit motion_design ou video_recording.",
      technical:
        "Classe en <code>motion_design</code> si la durée est strictement inférieure à 180 secondes et si la part de frames <code>footage</code> est strictement inférieure à 15 %. Sans frame footage, cette seconde condition est satisfaite.",
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
      action: "Détecte un plan filmé dominant.",
      technical:
        "Les embeddings DINO des frames footage sont normalisés en L2. Le produit scalaire calcule ensuite la similarité cosinus entre chaque paire. Pour chaque frame, on compte les voisines à ≥ 0,88 : celle qui en possède le plus devient le centre du cluster dominant. En cas d’égalité, la meilleure similarité moyenne l’emporte. Interview uniquement si ce cluster couvre au moins 50 % des frames footage et au moins 30 % de toutes les frames classifiées (footage + graphic + mixture).",
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
      action: "Extrait des frames JPEG timecodées.",
      technical:
        "FFmpeg échantillonne la vidéo toutes les 0,5 seconde et nomme chaque JPEG avec le timecode correspondant.",
      outputs: [artifact("outputs/images/*.jpg")],
    }),
    oneChild(
      box("Classifier les frames", {
        task: "frames.classify",
        phase: "inspection",
        phaseLabel: "Images",
        summary: "Classe footage, graphic ou mixture.",
        inputs: [artifact("outputs/images/*.jpg")],
        action: "Classe chaque frame par famille visuelle.",
        technical:
          "DINOv2 et CLIP produisent chacun un vecteur global par frame : 768 dimensions pour DINOv2 et 512 pour CLIP. Chaque vecteur est normalisé séparément en L2, puis les deux sont concaténés en un vecteur de 1 280 dimensions sans nouvelle normalisation L2 ; sa norme vaut donc environ √2. Le <code>StandardScaler</code> du pipeline Joblib standardise ensuite chaque feature avec les statistiques d’entraînement avant la régression logistique. Le modèle attend exactement ces 1 280 features pour classer l’image en footage, graphic ou mixture.",
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
      action: "Enregistre la route long_video.",
      technical:
        "Enregistre directement <code>long_video</code> dans le manifeste et ignore les étapes d’inspection visuelle, de classification des frames et d’OCR.",
    }),
    oneChild(
      box("Transcrire avec WhisperX", {
        task: "transcript.whisper",
        phase: "transcript",
        phaseLabel: "Audio",
        summary: "Transcrit et aligne l’audio sans diarisation.",
        inputs: [artifact("VIDEO_ID.mp4")],
        action: "Produit le transcript WhisperX brut timecodé.",
        technical:
          "Extrait l’audio en MP3, transcrit et aligne les paroles avec WhisperX, sans exécuter la diarisation Pyannote sur la route longue.",
        outputs: [
          artifact("outputs/transcripts_whisper/audio/VIDEO_ID.mp3"),
          artifact(
            "outputs/transcripts_whisper/transcript_1_brut.txt",
          ),
        ],
      }),
      oneChild(
        box("Créer le transcript plain depuis le brut", {
          task: "transcript.create_plain",
          phase: "transcript",
          phaseLabel: "Transcript",
          summary:
            "Produit le transcript plain directement depuis le transcript WhisperX brut.",
          inputs: [
            artifact(
              "outputs/transcripts_whisper/transcript_1_brut.txt",
            ),
          ],
          action: "Produit transcript_plain.txt depuis le brut.",
          technical:
            "Lit <code>transcript_1_brut.txt</code>, retire les timecodes et les éventuels labels <code>SPEAKER_XX</code>, puis concatène le texte restant dans <code>transcript_plain.txt</code>. Aucun transcript corrigé ou enrichi n’intervient sur la route longue.",
          outputs: [
            artifact(
              "outputs/transcripts_whisper/transcript_plain.txt",
            ),
          ],
        }),
        oneChild(
          box("Extraire les candidats speakers", {
            task: "speakers.propose",
            phase: "structure",
            phaseLabel: "Speakers",
            summary: "Repère localement les noms dans les auto-présentations.",
            inputs: [
              artifact(
                "outputs/transcripts_whisper/transcript_plain.txt",
              ),
            ],
            action: "Extrait les candidats depuis le transcript plain.",
            technical:
              "Analyse tout <code>transcript_plain.txt</code> pour extraire les noms qui suivent « je m’appelle », « je suis » ou « moi c’est », puis écrit ces candidats dans <code>speaker_candidates.json</code> sans appeler le LLM. Le titre et les 1 000 premiers caractères sont seulement stockés dans cet artefact ; ils seront envoyés par l’étape de validation suivante.",
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
              action: "Envoie le contexte au LLM et valide les speakers.",
              llmCall:
                "Le titre de la vidéo, les 1 000 premiers caractères du transcript et les candidats extraits dans le transcript complet. Aucun texte OCR n’est disponible sur la route longue.",
              tooltipLabel: "Voir le prompt de validation des speakers",
              tooltip:
                "<span class=\"info-tooltip-example prompt-only\"><strong>Exemple simplifié du prompt</strong><code><b>SYSTEM</b> — Garde uniquement les personnes physiques trouvées dans le titre, le transcript ou les candidats. Pour chacune, retourne <b>speaker</b> et <b>title</b>.<br><br><b>USER</b> — Identifie les speakers dans ces sources.<br>{<br>&nbsp;&nbsp;&quot;video_title&quot;: &quot;Portrait de Marc Dupont — Example Corp&quot;,<br>&nbsp;&nbsp;&quot;transcript_excerpt&quot;: &quot;Bonjour, je suis Marc Dupont, responsable innovation chez Example Corp…&quot;,<br>&nbsp;&nbsp;&quot;ocr_detected_texts&quot;: [],<br>&nbsp;&nbsp;&quot;candidates&quot;: [{&quot;name&quot;: &quot;Marc Dupont&quot;}]<br>}</code></span>",
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
                action: "Crée les chunks de niveau detail.",
                technical:
                  "Découpe le transcript plain par phrases en passages d’environ 1 000 caractères et les écrit au niveau <code>detail</code>.",
                outputs: [
                  artifact(
                    "outputs/chunks/transcript_chunks.json",
                    "niveau detail",
                  ),
                ],
              }),
              oneChild(
                box("Créer les chunks section", {
                  task: "chunks.summarize_sections",
                  phase: "structure",
                  phaseLabel: "Chunks",
                  llm: true,
                  summary: "Résume chaque groupe de 6 chunks détail.",
                  inputs: [
                    artifact("outputs/chunks/transcript_chunks.json"),
                  ],
                  action: "Crée les chunks de niveau section.",
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
                    ${box("Créer le chunk global", {
                      task: "chunks.summarize_video",
                      phase: "structure",
                      phaseLabel: "Données finales",
                      kind: "result",
                      llm: true,
                      summary: "Résume les sections en un chunk global.",
                      inputs: [
                        artifact("outputs/chunks/transcript_chunks.json"),
                      ],
                      action: "Crée le chunk global et la hiérarchie.",
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
      action: "Choisit la route courte ou longue.",
      technical:
        "Lit <code>duration_seconds</code> dans les métadonnées YouTube. Le seuil est strict : une durée supérieure à 600 secondes prend la route longue ; 600 secondes exactement reste sur la route courte.",
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
      action: "Initialise le contexte et le plan d’exécution.",
      technical:
        "Sonde le fichier vidéo, charge les métadonnées et le manifeste, construit le plan d’exécution puis checkpoint le manifeste avant et après chaque tâche.",
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
