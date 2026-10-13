// Generated from the Phoenix traces referenced by each scenario (see app.js).
// Each step's span output, shortened: long strings, lists and vectors are cut.
const recordedStepOutputs = {
 "aymerich": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": true,
    "ambiguous": false,
    "requested": [
     "Aymerich"
    ],
    "suggestion_speakers": [],
    "suggestion_transcripts": [
     {
      "person": "Aymerich",
      "score": 1.0
     }
    ]
   },
   "company_resolution": {
    "requested": [],
    "suggestion_companies": []
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "sql_search",
   "analytics_requested": false,
   "raw_question": "Qui est Aymerich ?",
   "query_text": "Qui est Aymerich ?",
   "query_text_bm25": "Aymerich identité",
   "title_hints": [],
   "persons": [
    "Aymerich"
   ],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": false,
   "transcription_requested": false,
   "top_k": null,
   "final_k": null
  },
  "sql_search": {
   "mode": "specific_persons",
   "entity_queries": [
    {
     "entity_type": "transcript",
     "entity": "Aymerich",
     "result_count": 2,
     "lookup_strategy": "transcript_fallback"
    }
   ],
   "result_count": 2,
   "deduplication": {
    "input_count": 2,
    "duplicate_count": 0,
    "output_count": 2,
    "key": "video_id"
   }
  },
  "lookup": {
   "person_in_transcript_lookup": {
    "result_count": 2,
    "lookup_strategy": "transcript_fallback"
   },
   "deduplicate_videos": {
    "input_count": 2,
    "duplicate_count": 0,
    "output_count": 2,
    "key": "video_id"
   }
  },
  "generate": {
   "answer": "Aymerich Makem est un alternant qui a été co-encadré par Stanislas Blanchard (Energy Manager chez Novares) et accompagné par Loïc Maréchal (Directeur HSE chez Novares). Il a travaillé sur des projets liés au management de l'énergie, comme la création d'une base de données des con…",
   "action": "answer",
   "source_indexes": [
    1,
    2
   ]
  },
  "persist": {
   "conversation_id": 343,
   "message_id": 441,
   "memory": {
    "current_topic": "Aymerich",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": [
     "Témoignage Tuteur : Group Energy Manager, Novares",
     "Témoignage Tuteur : Loïc Maréchal, Directeur HSE, Novares"
    ]
   }
  },
  "end": {
   "action": "answer",
   "answer": "Aymerich Makem est un alternant qui a été co-encadré par Stanislas Blanchard (Energy Manager chez Novares) et accompagné par Loïc Maréchal (Directeur HSE chez Novares). Il a travaillé sur des projets liés au management de l'énergie, comme la création d'une base de données des con…",
   "sources": [
    "Témoignage Tuteur : Group Energy Manager, Novares",
    "Témoignage Tuteur : Loïc Maréchal, Directeur HSE, Novares"
   ]
  }
 },
 "salim": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": true,
    "ambiguous": false,
    "requested": [
     "Salim"
    ],
    "suggestion_speakers": [
     {
      "person": "Salim Announ",
      "score": 1.0
     },
     {
      "person": "Halima Ahlem Saim",
      "score": 0.889
     }
    ],
    "suggestion_transcripts": [
     {
      "person": "Salim",
      "score": 1.0
     }
    ]
   },
   "company_resolution": {
    "requested": [],
    "suggestion_companies": []
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "sql_search",
   "analytics_requested": false,
   "raw_question": "J'ai besoin de la transcription de la vidéo de Salim.",
   "query_text": "Quelle est la transcription de la vidéo de Salim ?",
   "query_text_bm25": "transcription vidéo Salim",
   "title_hints": [],
   "persons": [
    "Halima Ahlem Saim",
    "Salim Announ",
    "Salim"
   ],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": false,
   "transcription_requested": true,
   "top_k": null,
   "final_k": null
  },
  "sql_search": {
   "mode": "transcript_verbatim",
   "sql": "SELECT v.id, v.title, v.url, v.thumbnail_medium_url, t.transcript AS document_text, ARRAY( SELECT person_row.name FROM video_speakers video_person JOIN speakers person_row ON person_row.id = video_person.speaker_id WHERE video_person.video_id = v.id ORDER BY person_row.id ) , ts_…",
   "params": [
    "transcription vidéo Salim",
    [
     "Témoignage Tuteur : Salim Announ, Chargé de Projet Territorial, Roche",
     "Portrait d'Halima Ahlem Saim"
    ]
   ],
   "result_count": 1
  },
  "lookup": {
   "person_in_speaker_lookup": {
    "result_count": 2,
    "lookup_strategy": "transcript_fallback"
   },
   "person_in_transcript_lookup": {
    "result_count": 1,
    "lookup_strategy": "transcript_fallback"
   },
   "deduplicate_videos": {
    "input_count": 3,
    "duplicate_count": 1,
    "output_count": 2,
    "key": "video_id"
   }
  },
  "format": {
   "answer": "Voici la transcription :\n\nBonjour, je m'appelle Salim Announ, je suis chargé de projet territorial au sein de la société Roche Diagnostics et j'ai eu le plaisir d'accompagner Kilian Fromilag depuis septembre 2023. Mon métier consiste à faire rayonner la marque d'autosurveillance…"
  },
  "persist": {
   "conversation_id": 271,
   "message_id": 351,
   "memory": {
    "current_topic": "transcription de la vidéo de Salim",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": [
     "Témoignage Tuteur : Salim Announ, Chargé de Projet Territorial, Roche"
    ]
   }
  },
  "end": {
   "action": "answer",
   "answer": "Voici la transcription :\n\nBonjour, je m'appelle Salim Announ, je suis chargé de projet territorial au sein de la société Roche Diagnostics et j'ai eu le plaisir d'accompagner Kilian Fromilag depuis septembre 2023. Mon métier consiste à faire rayonner la marque d'autosurveillance…",
   "sources": [
    "Témoignage Tuteur : Salim Announ, Chargé de Projet Territorial, Roche"
   ]
  }
 },
 "views": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": false,
    "ambiguous": false,
    "requested": [],
    "suggestion_speakers": [],
    "suggestion_transcripts": [],
    "reason": "no_planned_persons"
   },
   "company_resolution": {
    "requested": [],
    "suggestion_companies": []
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "sql_search",
   "analytics_requested": true,
   "raw_question": "Quelle vidéo a le plus de vues ?",
   "query_text": "Quelle vidéo a le plus de vues ?",
   "query_text_bm25": "vidéo plus vues",
   "title_hints": [],
   "persons": [],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": false,
   "transcription_requested": false,
   "top_k": null,
   "final_k": null
  },
  "sql_search": {
   "mode": "analytics",
   "strategy": "llm_text_to_sql",
   "sql": "SELECT v.id AS video_id, v.title AS video_title, v.url AS video_url, v.thumbnail_medium_url, s.snapshot_date AS stats_snapshot_date, s.view_count FROM videos v JOIN LATERAL (SELECT snapshot_date, view_count FROM stats WHERE video_id = v.id ORDER BY snapshot_date DESC, data_collec…",
   "params": [],
   "result_count": 1,
   "result_intro": "Voici la vidéo avec le plus de vues :",
   "validation": {
    "valid": true,
    "errors": [],
    "relations": [
     "stats",
     "videos"
    ],
    "ctes": [],
    "placeholder_count": 0,
    "param_count": 0
   },
   "cost_validation": {
    "valid": true,
    "total_cost": 1891.29,
    "max_total_cost": 100000.0
   },
   "status": "executed",
   "execution": {
    "bounded_sql": "SELECT * FROM ( SELECT v.id AS video_id, v.title AS video_title, v.url AS video_url, v.thumbnail_medium_url, s.snapshot_date AS stats_snapshot_date, s.view_count FROM videos v JOIN LATERAL (SELECT snapshot_date, view_count FROM stats WHERE video_id = v.id ORDER BY snapshot_date D…",
    "columns": [
     "video_id",
     "video_title",
     "video_url",
     "… +3 autres"
    ],
    "row_count": 1
   }
  },
  "analytics": {
   "sql_generation": {
    "status": "generated",
    "sql": "SELECT v.id AS video_id, v.title AS video_title, v.url AS video_url, v.thumbnail_medium_url, s.snapshot_date AS stats_snapshot_date, s.view_count FROM videos v JOIN LATERAL (SELECT snapshot_date, view_count FROM stats WHERE video_id = v.id ORDER BY snapshot_date DESC, data_collec…",
    "result_intro": "Voici la vidéo avec le plus de vues :"
   },
   "sql_validation": {
    "valid": true,
    "errors": [],
    "relations": [
     "stats",
     "videos"
    ],
    "ctes": [],
    "placeholder_count": 0,
    "param_count": 0
   },
   "sql_cost_validation": {
    "valid": true,
    "total_cost": 1891.29,
    "max_total_cost": 100000.0
   },
   "sql_execution": {
    "columns": [
     "video_id",
     "video_title",
     "video_url",
     "… +3 autres"
    ],
    "row_count": 1,
    "results": [
     {
      "chunk_id": 128,
      "video_title": "Portrait d'Hugo Gérardin - Promotion 2021",
      "video_url": "https://www.youtube.com/watch?v=n2RLhYOfk-E",
      "thumbnail_medium_url": "https://i.ytimg.com/vi/n2RLhYOfk-E/mqdefault.jpg",
      "chunk_index": 0,
      "text": "video_id: 128\nvideo_title: \"Portrait d'Hugo Gérardin - Promotion 2021\"\nvideo_url: \"https://www.youtube.com/watch?v=n2RLhYOfk-E\"\nthumbnail_medium_url: \"https://i.ytimg.com/vi/n2RLhYOfk-E/mqdefault.jpg\"\nstats_snapshot_date: \"2026-09-13\"\nview_count: 21142",
      "persons": [],
      "bm25_score": null,
      "result_intro": "Voici la vidéo avec le plus de vues :"
     }
    ]
   }
  },
  "generate": {
   "answer": "La vidéo avec le plus de vues est **« Portrait d'Hugo Gérardin - Promotion 2021 »** (URL : [lien](https://www.youtube.com/watch?v=n2RLhYOfk-E)), avec **21 142 vues** au **13 septembre 2026**.",
   "action": "answer",
   "source_indexes": [
    1
   ]
  },
  "persist": {
   "conversation_id": 336,
   "message_id": 434,
   "memory": {
    "current_topic": "vidéos",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": [
     "Portrait d'Hugo Gérardin - Promotion 2021"
    ]
   }
  },
  "end": {
   "action": "answer",
   "answer": "La vidéo avec le plus de vues est **« Portrait d'Hugo Gérardin - Promotion 2021 »** (URL : [lien](https://www.youtube.com/watch?v=n2RLhYOfk-E)), avec **21 142 vues** au **13 septembre 2026**.",
   "sources": [
    "Portrait d'Hugo Gérardin - Promotion 2021"
   ]
  }
 },
 "views_growth": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": false,
    "ambiguous": false,
    "requested": [],
    "suggestion_speakers": [],
    "suggestion_transcripts": [],
    "reason": "no_planned_persons"
   },
   "company_resolution": {
    "requested": [],
    "suggestion_companies": []
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "sql_search",
   "analytics_requested": true,
   "raw_question": "Quelle vidéo a gagné le plus de vues entre août et septembre ?",
   "query_text": "Quelle vidéo a gagné le plus de vues entre août et septembre ?",
   "query_text_bm25": "vidéo plus vues août septembre",
   "title_hints": [],
   "persons": [],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": false,
   "transcription_requested": false,
   "top_k": null,
   "final_k": null
  },
  "sql_search": {
   "mode": "analytics",
   "strategy": "llm_text_to_sql",
   "sql": "SELECT v.id AS video_id, v.title AS video_title, v.url AS video_url, v.thumbnail_medium_url AS thumbnail_medium_url, s_aug.snapshot_date AS stats_snapshot_date_aug, s_aug.view_count AS view_count_aug, s_sep.snapshot_date AS stats_snapshot_date_sep, s_sep.view_count AS view_count_…",
   "params": [
    "2026-08-01",
    "2026-09-01",
    "2026-09-01",
    "… +1 autres"
   ],
   "result_count": 1,
   "result_intro": "Voici la vidéo ayant gagné le plus de vues entre août et septembre :",
   "validation": {
    "valid": true,
    "errors": [],
    "relations": [
     "stats",
     "videos"
    ],
    "ctes": [],
    "placeholder_count": 4,
    "param_count": 4
   },
   "cost_validation": {
    "valid": true,
    "total_cost": 3433.35,
    "max_total_cost": 100000.0
   },
   "status": "executed",
   "execution": {
    "bounded_sql": "SELECT * FROM ( SELECT v.id AS video_id, v.title AS video_title, v.url AS video_url, v.thumbnail_medium_url AS thumbnail_medium_url, s_aug.snapshot_date AS stats_snapshot_date_aug, s_aug.view_count AS view_count_aug, s_sep.snapshot_date AS stats_snapshot_date_sep, s_sep.view_coun…",
    "columns": [
     "video_id",
     "video_title",
     "video_url",
     "… +6 autres"
    ],
    "row_count": 1
   }
  },
  "analytics": {
   "sql_generation": {
    "status": "generated",
    "sql": "SELECT v.id AS video_id, v.title AS video_title, v.url AS video_url, v.thumbnail_medium_url AS thumbnail_medium_url, s_aug.snapshot_date AS stats_snapshot_date_aug, s_aug.view_count AS view_count_aug, s_sep.snapshot_date AS stats_snapshot_date_sep, s_sep.view_count AS view_count_…",
    "result_intro": "Voici la vidéo ayant gagné le plus de vues entre août et septembre :"
   },
   "sql_validation": {
    "valid": true,
    "errors": [],
    "relations": [
     "stats",
     "videos"
    ],
    "ctes": [],
    "placeholder_count": 4,
    "param_count": 4
   },
   "sql_cost_validation": {
    "valid": true,
    "total_cost": 3433.35,
    "max_total_cost": 100000.0
   },
   "sql_execution": {
    "columns": [
     "video_id",
     "video_title",
     "video_url",
     "… +6 autres"
    ],
    "row_count": 1,
    "results": [
     {
      "chunk_id": 23,
      "video_title": "Bastien Roche, Chargé de Projet rénovation énergétique, Effy",
      "video_url": "https://www.youtube.com/watch?v=8_eTP4xhZmk",
      "thumbnail_medium_url": "https://i.ytimg.com/vi/8_eTP4xhZmk/mqdefault.jpg",
      "chunk_index": 0,
      "text": "video_id: 23\nvideo_title: \"Bastien Roche, Chargé de Projet rénovation énergétique, Effy\"\nvideo_url: \"https://www.youtube.com/watch?v=8_eTP4xhZmk\"\nthumbnail_medium_url: \"https://i.ytimg.com/vi/8_eTP4xhZmk/mqdefault.jpg\"\nstats_snapshot_date_aug: \"2026-08-28\"\nview_count_aug: 131\nsta…",
      "persons": [],
      "bm25_score": null,
      "result_intro": "Voici la vidéo ayant gagné le plus de vues entre août et septembre :"
     }
    ]
   }
  },
  "generate": {
   "answer": "La vidéo ayant gagné le plus de vues entre août et septembre est **« Bastien Roche, Chargé de Projet rénovation énergétique, Effy »** (URL : [lien](https://www.youtube.com/watch?v=8_eTP4xhZmk)), avec un gain de **12 vues** entre le 28 août 2026 et le 13 septembre 2026.",
   "action": "answer",
   "source_indexes": [
    1
   ]
  },
  "persist": {
   "conversation_id": 337,
   "message_id": 435,
   "memory": {
    "current_topic": "vidéos et vues entre août et septembre",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": [
     "Bastien Roche, Chargé de Projet rénovation énergétique, Effy"
    ]
   }
  },
  "end": {
   "action": "answer",
   "answer": "La vidéo ayant gagné le plus de vues entre août et septembre est **« Bastien Roche, Chargé de Projet rénovation énergétique, Effy »** (URL : [lien](https://www.youtube.com/watch?v=8_eTP4xhZmk)), avec un gain de **12 vues** entre le 28 août 2026 et le 13 septembre 2026.",
   "sources": [
    "Bastien Roche, Chargé de Projet rénovation énergétique, Effy"
   ]
  }
 },
 "gaelle": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": true,
    "ambiguous": false,
    "requested": [
     "Gaëlle"
    ],
    "suggestion_speakers": [
     {
      "person": "Gaelle Lecomte",
      "score": 1.0
     }
    ],
    "suggestion_transcripts": [
     {
      "person": "Gaëlle",
      "score": 1.0
     }
    ]
   },
   "company_resolution": {
    "requested": [],
    "suggestion_companies": []
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "sql_search",
   "analytics_requested": false,
   "raw_question": "J'ai besoin de la description de la vidéo de Gaëlle.",
   "query_text": "Quelle est la description de la vidéo de Gaëlle ?",
   "query_text_bm25": "description vidéo Gaëlle",
   "title_hints": [],
   "persons": [
    "Gaelle Lecomte",
    "Gaëlle"
   ],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": true,
   "transcription_requested": false,
   "top_k": null,
   "final_k": null
  },
  "sql_search": {
   "mode": "description",
   "sql": "SELECT v.id, v.title, v.url, v.thumbnail_medium_url, v.description FROM videos v WHERE btrim(regexp_replace(unaccent(lower(coalesce(v.title, ''))), '[^[:alnum:]]+', ' ', 'g')) LIKE ANY(ARRAY(SELECT concat(chr(37), btrim(regexp_replace(unaccent(lower(coalesce(title_hint, ''))), '[…",
   "params": [
    [
     "📢 Parole d'Alumni : Gaelle Lecomte, Chef de Projet Web & Digital, Pierre Cattier"
    ]
   ],
   "result_count": 1
  },
  "lookup": {
   "person_in_speaker_lookup": {
    "result_count": 1,
    "lookup_strategy": "transcript_fallback"
   },
   "person_in_transcript_lookup": {
    "result_count": 1,
    "lookup_strategy": "transcript_fallback"
   },
   "deduplicate_videos": {
    "input_count": 2,
    "duplicate_count": 1,
    "output_count": 1,
    "key": "video_id"
   }
  },
  "format": {
   "answer": "Voici la description :\n\nTitre: 📢 Parole d'Alumni : Gaelle Lecomte, Chef de Projet Web & Digital, Pierre Cattier\nURL: https://www.youtube.com/watch?v=w1zV0-i8he8\nDescription: 🎬 Interview de Gaëlle Le Comte, Chef de Projet Web & Digital, Pierre Cattier, Alumni du MSc Biotechnologie…"
  },
  "persist": {
   "conversation_id": 273,
   "message_id": 354,
   "memory": {
    "current_topic": "description de la vidéo de Gaëlle",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": [
     "📢 Parole d'Alumni : Gaelle Lecomte, Chef de Projet Web & Digital, Pierre Cattier"
    ]
   }
  },
  "end": {
   "action": "answer",
   "answer": "Voici la description :\n\nTitre: 📢 Parole d'Alumni : Gaelle Lecomte, Chef de Projet Web & Digital, Pierre Cattier\nURL: https://www.youtube.com/watch?v=w1zV0-i8he8\nDescription: 🎬 Interview de Gaëlle Le Comte, Chef de Projet Web & Digital, Pierre Cattier, Alumni du MSc Biotechnologie…",
   "sources": [
    "📢 Parole d'Alumni : Gaelle Lecomte, Chef de Projet Web & Digital, Pierre Cattier"
   ]
  }
 },
 "comparison": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": true,
    "ambiguous": false,
    "requested": [
     "Sophie Ollivier",
     "Chloé Leprètre"
    ],
    "suggestion_speakers": [
     {
      "person": "Sophie Ollivier",
      "score": 1.0
     },
     {
      "person": "Chloé Lepraître",
      "score": 0.912
     }
    ],
    "suggestion_transcripts": [
     {
      "person": "Sophie Ollivier",
      "score": 1.0
     },
     {
      "person": "Chloé Leprètre",
      "score": 1.0
     }
    ]
   },
   "company_resolution": {
    "requested": [],
    "suggestion_companies": []
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "sql_search",
   "analytics_requested": false,
   "raw_question": "Des points communs entre Sophie Ollivier et Chloé Leprètre ?",
   "query_text": "Quels sont les points communs entre Sophie Ollivier et Chloé Leprètre ?",
   "query_text_bm25": "points communs Sophie Ollivier Chloé Leprètre",
   "title_hints": [],
   "persons": [
    "Sophie Ollivier",
    "Chloé Lepraître",
    "Chloé Leprètre"
   ],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": false,
   "transcription_requested": false,
   "top_k": null,
   "final_k": null
  },
  "sql_search": {
   "mode": "specific_persons",
   "entity_queries": [
    {
     "entity_type": "speaker",
     "entity": "Sophie Ollivier",
     "result_count": 1,
     "lookup_strategy": "persons_table"
    },
    {
     "entity_type": "speaker",
     "entity": "Chloé Lepraître",
     "result_count": 1,
     "lookup_strategy": "persons_table"
    },
    {
     "entity_type": "transcript",
     "entity": "Sophie Ollivier",
     "result_count": 1,
     "lookup_strategy": "transcript_fallback"
    },
    "… +1 autres"
   ],
   "result_count": 2,
   "deduplication": {
    "input_count": 4,
    "duplicate_count": 2,
    "output_count": 2,
    "key": "video_id"
   }
  },
  "lookup": {
   "person_in_speaker_lookup": {
    "result_count": 1,
    "lookup_strategy": "persons_table"
   },
   "person_in_transcript_lookup": {
    "result_count": 1,
    "lookup_strategy": "transcript_fallback"
   },
   "deduplicate_videos": {
    "input_count": 4,
    "duplicate_count": 2,
    "output_count": 2,
    "key": "video_id"
   }
  },
  "generate": {
   "answer": "Sophie Ollivier et Chloé Leprètre partagent plusieurs points communs :\n\n- **Employeur** : Elles travaillent toutes les deux pour les **laboratoires Servier**, dans le domaine de la **recherche et développement (R&D)**.\n- **Secteur d'activité** : Leur travail est centré sur l'**in…",
   "action": "answer",
   "source_indexes": [
    1,
    2
   ]
  },
  "persist": {
   "conversation_id": 110,
   "message_id": 187,
   "memory": {
    "current_topic": "points communs entre Sophie Ollivier et Chloé Leprètre",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": [
     "Sophie Ollivier, Chief Data Officer R&D, Servier",
     "Chloé Leprètre, Head of R&D Search & Evaluation chez Servier"
    ]
   }
  },
  "end": {
   "action": "answer",
   "answer": "Sophie Ollivier et Chloé Leprètre partagent plusieurs points communs :\n\n- **Employeur** : Elles travaillent toutes les deux pour les **laboratoires Servier**, dans le domaine de la **recherche et développement (R&D)**.\n- **Secteur d'activité** : Leur travail est centré sur l'**in…",
   "sources": [
    "Sophie Ollivier, Chief Data Officer R&D, Servier",
    "Chloé Leprètre, Head of R&D Search & Evaluation chez Servier"
   ]
  }
 },
 "ionis": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": false,
    "ambiguous": false,
    "requested": [],
    "suggestion_speakers": [],
    "suggestion_transcripts": [],
    "reason": "no_planned_persons"
   },
   "company_resolution": {
    "requested": [
     "Ionis-STM"
    ],
    "suggestion_companies": [
     {
      "company": "ionis stm",
      "score": 1.0
     },
     {
      "company": "lonis stm",
      "score": 0.889
     }
    ]
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "vector_search",
   "analytics_requested": false,
   "raw_question": "Pourquoi faire Ionis-STM ?",
   "query_text": "Pourquoi choisir de faire des études à Ionis-STM ?",
   "query_text_bm25": "études Ionis-STM raisons choix",
   "title_hints": [],
   "persons": [],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": false,
   "transcription_requested": false,
   "top_k": 40,
   "final_k": 5
  },
  "vector_search": {
   "prefilter": {
    "sql": null,
    "params": [],
    "applied": false,
    "sql_prefilters": false,
    "candidate_count": null,
    "candidate_chunk_ids": null,
    "general_question_only": true
   }
  },
  "retrieval": {
   "bm25": {
    "result_count": 1
   },
   "vector": {
    "result_count": 40
   },
   "rrf": {
    "rank_constant": 60,
    "bm25_count": 1,
    "vector_count": 40,
    "fused_count": 30
   },
   "rerank": {
    "applied": true,
    "input_count": 30,
    "output_count": 5,
    "relevance_scores": [
     0.9306097,
     0.93035704,
     0.93035704,
     "… +2 autres"
    ]
   },
   "passages": [
    {
     "video_title": "Ionis-STM : Remise de diplomes Promo 2011",
     "text": "Si un étudiant a du mal sur un projet, il y en aura toujours deux ou trois qui vont pouvoir l'aider, l'appuyer. On a toujours travaillé ensemble et on s'est toujours soutenus. Et même maintenant, je pense que c'est des amis. De toute façon, on reste en contact, même maintenant qu…"
    },
    {
     "video_title": "Je me projette en tant que Chef de Projet Éolien - Hadil Bekirane",
     "text": "Bonjour, je m'appelle Hadil Bekirane, je suis étudiante en Master of Science Énergie et Management, spécialisation Ingénierie et Développement, en alternance en tant que chargée de prospection éolien chez ENERTRAG France. J'ai choisi Ionis-STM pour plusieurs raisons, la réputatio…"
    },
    {
     "video_title": "Maëva Lima, Assistante Marketing International, Talika",
     "text": "Avant d'intégrer Ionis-STM, j'ai d'abord commencé par une première année commune aux études de santé, donc une année de PACES, et ensuite j'ai enchaîné sur une licence en biologie et chimie. J'ai choisi Ionis-STM parce qu'à la base mes études étaient beaucoup axées sur les scienc…"
    },
    "… +2 autres"
   ]
  },
  "generate": {
   "answer": "Choisir Ionis-STM permet d’acquérir une **double compétence** (scientifique et managériale), adaptée aux besoins des entreprises et aux nouveaux business models. \n\nLes points forts incluent :\n- Une **pédagogie par projets concrets** et cas réels, favorisant l’apprentissage pratiq…",
   "action": "answer",
   "source_indexes": [
    1,
    2,
    3,
    "… +2 autres"
   ]
  },
  "persist": {
   "conversation_id": 338,
   "message_id": 436,
   "memory": {
    "current_topic": "choix des études à Ionis-STM",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": [
     "Ionis-STM : Remise de diplomes Promo 2011",
     "Je me projette en tant que Chef de Projet Éolien - Hadil Bekirane",
     "Maëva Lima, Assistante Marketing International, Talika",
     "… +2 autres"
    ]
   }
  },
  "end": {
   "action": "answer",
   "answer": "Choisir Ionis-STM permet d’acquérir une **double compétence** (scientifique et managériale), adaptée aux besoins des entreprises et aux nouveaux business models. \n\nLes points forts incluent :\n- Une **pédagogie par projets concrets** et cas réels, favorisant l’apprentissage pratiq…",
   "sources": [
    "Ionis-STM : Remise de diplomes Promo 2011",
    "Je me projette en tant que Chef de Projet Éolien - Hadil Bekirane",
    "Maëva Lima, Assistante Marketing International, Talika",
    "… +2 autres"
   ]
  }
 },
 "ocean": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": false,
    "ambiguous": false,
    "requested": [],
    "suggestion_speakers": [],
    "suggestion_transcripts": [],
    "reason": "no_planned_persons"
   },
   "company_resolution": {
    "requested": [],
    "suggestion_companies": []
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "vector_search",
   "analytics_requested": false,
   "analytics_scope": null,
   "analytics_metric": null,
   "analytics_order": null,
   "analytics_rank_start": null,
   "analytics_rank_end": null,
   "raw_question": "C'est quoi l'océan rouge ?",
   "query_text": "Qu'est-ce que l'océan rouge ?",
   "query_text_bm25": "océan rouge définition",
   "title_hints": [],
   "persons": [],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": false,
   "transcription_requested": false,
   "top_k": 40,
   "final_k": 5
  },
  "vector_search": {
   "prefilter": {
    "sql": null,
    "params": [],
    "applied": false,
    "sql_prefilters": false,
    "candidate_count": null,
    "candidate_chunk_ids": null,
    "general_question_only": true
   }
  },
  "retrieval": {
   "bm25": {
    "result_count": 0
   },
   "vector": {
    "result_count": 40
   },
   "rrf": {
    "rank_constant": 60,
    "bm25_count": 0,
    "vector_count": 40,
    "fused_count": 30
   },
   "rerank": {
    "applied": true,
    "input_count": 30,
    "output_count": 5,
    "relevance_scores": [
     0.7646105,
     0.26102942,
     0.21840255,
     "… +2 autres"
    ]
   },
   "passages": [
    {
     "video_title": "Stéphane Carpentier, intervenant en stratégie \"Océan Bleu\", Executive MBA",
     "text": "L'Océan rouge, c'est un marché saturé de concurrents, tout simplement, où les acteurs essaient d'être leaders sur leur marché, à la différence d'une stratégie Océan Bleu, donc un Océan Bleu où on met sa concurrence hors-jeu. Comme on le dit très souvent aujourd'hui, les stratégie…"
    },
    {
     "video_title": "Stéphane Carpentier, intervenant en stratégie \"Océan Bleu\", Executive MBA",
     "text": "On a aussi la friteuse Seb qui permet de faire des frites avec moins de matières grasses, ça c'est une belle stratégie Océan bleu qui a révolutionné ce monde de la cuisine. Beaucoup d'entreprises ont appliqué cette technique qui amène beaucoup de profit, et représente une innovat…"
    },
    {
     "video_title": "Webinar - « L’énergie, des métiers d’avenir. Pourquoi pas vous ? »",
     "text": "Mais ce qu'il faut voir, c'est que les objectifs sont clairement d'augmenter la part du renouvelable dans les énergies de demain. Alors là, j'ai pris un exemple, je crois, ça c'est l'Allemagne. Donc, eux, ils ont une proportion un peu plus faible de nucléaire, qui est en bleu, ma…"
    },
    "… +2 autres"
   ]
  },
  "generate": {
   "answer": "L'**océan rouge** désigne un marché **saturé de concurrents**, où les acteurs se livrent une bataille acharnée pour dominer le secteur. À l'inverse, une stratégie *océan bleu* vise à créer un nouvel espace marché, sans concurrence directe, en innovant de manière radicale.",
   "action": "answer",
   "source_indexes": [
    1
   ]
  },
  "persist": {
   "conversation_id": 97,
   "message_id": 170,
   "memory": {
    "current_topic": "océan rouge",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": [
     "Stéphane Carpentier, intervenant en stratégie \"Océan Bleu\", Executive MBA"
    ]
   }
  },
  "end": {
   "action": "answer",
   "answer": "L'**océan rouge** désigne un marché **saturé de concurrents**, où les acteurs se livrent une bataille acharnée pour dominer le secteur. À l'inverse, une stratégie *océan bleu* vise à créer un nouvel espace marché, sans concurrence directe, en innovant de manière radicale.",
   "sources": [
    "Stéphane Carpentier, intervenant en stratégie \"Océan Bleu\", Executive MBA"
   ]
  }
 },
 "fadila": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": true,
    "ambiguous": false,
    "requested": [
     "Fadila"
    ],
    "suggestion_speakers": [
     {
      "person": "Fadila Ouro Sama",
      "score": 1.0
     }
    ],
    "suggestion_transcripts": [
     {
      "person": "Fadila",
      "score": 1.0
     }
    ]
   },
   "company_resolution": {
    "requested": [],
    "suggestion_companies": []
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "sql_search",
   "analytics_requested": false,
   "raw_question": "Quelles questions ont été posées à Fadila ?",
   "query_text": "Quelles questions ont été posées à Fadila ?",
   "query_text_bm25": "questions posées Fadila",
   "title_hints": [],
   "persons": [
    "Fadila Ouro Sama",
    "Fadila"
   ],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": false,
   "transcription_requested": false,
   "top_k": null,
   "final_k": null
  },
  "sql_search": {
   "mode": "specific_persons",
   "entity_queries": [
    {
     "entity_type": "speaker",
     "entity": "Fadila Ouro Sama",
     "result_count": 1,
     "lookup_strategy": "persons_table"
    },
    {
     "entity_type": "transcript",
     "entity": "Fadila",
     "result_count": 1,
     "lookup_strategy": "transcript_fallback"
    }
   ],
   "result_count": 1,
   "deduplication": {
    "input_count": 2,
    "duplicate_count": 1,
    "output_count": 1,
    "key": "video_id"
   }
  },
  "lookup": {
   "person_in_speaker_lookup": {
    "result_count": 1,
    "lookup_strategy": "persons_table"
   },
   "person_in_transcript_lookup": {
    "result_count": 1,
    "lookup_strategy": "transcript_fallback"
   },
   "deduplicate_videos": {
    "input_count": 2,
    "duplicate_count": 1,
    "output_count": 1,
    "key": "video_id"
   }
  },
  "generate": {
   "answer": "Les questions posées à Fadila dans l'interview sont les suivantes :\n\n- Bonjour, comment tu t’appelles ?\n- Tu es en alternance actuellement ?\n- Tu as fait quelles études avant d’intégrer Ionis-STM ?\n- Pourquoi tu as choisi Ionis-STM ?\n- Qu’est-ce qui te plaît dans ta formation ?\n-…",
   "action": "answer",
   "source_indexes": [
    1
   ]
  },
  "persist": {
   "conversation_id": 344,
   "message_id": 443,
   "memory": {
    "current_topic": "questions posées à Fadila",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": [
     "Devenir Responsable Marketing Cosmétiques – Fadila, Chef de Produits Marketing, HUYGENS"
    ]
   }
  },
  "end": {
   "action": "answer",
   "answer": "Les questions posées à Fadila dans l'interview sont les suivantes :\n\n- Bonjour, comment tu t’appelles ?\n- Tu es en alternance actuellement ?\n- Tu as fait quelles études avant d’intégrer Ionis-STM ?\n- Pourquoi tu as choisi Ionis-STM ?\n- Qu’est-ce qui te plaît dans ta formation ?\n-…",
   "sources": [
    "Devenir Responsable Marketing Cosmétiques – Fadila, Chef de Produits Marketing, HUYGENS"
   ]
  }
 },
 "laura": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "annex_direct": {
   "route": "annex_direct",
   "annex_question": "Qui est Laura Tyan ?",
   "person_resolution": {
    "applied": false,
    "ambiguous": false,
    "requested": [
     "Laura Tyan"
    ],
    "reason": "local_annex_skip_remaining_pipeline"
   }
  }
 },
 "australie": {
  "initialize": {
   "reformulation_model": "mistral-medium-latest",
   "planner_model": "mistral-medium-latest",
   "analytics_sql_model": "mistral-medium-latest",
   "answer_model": "mistral-medium-latest"
  },
  "resolve_entities": {
   "person_resolution": {
    "applied": false,
    "ambiguous": false,
    "requested": [],
    "suggestion_speakers": [],
    "suggestion_transcripts": [],
    "reason": "no_planned_persons"
   },
   "company_resolution": {
    "requested": [],
    "suggestion_companies": []
   },
   "title_resolution": {
    "requested": null,
    "suggestion_titles": []
   }
  },
  "build_execution_plan": {
   "route": "vector_search",
   "analytics_requested": false,
   "raw_question": "Quelle est la capitale de l'Australie ?",
   "query_text": "Quelle est la capitale de l'Australie ?",
   "query_text_bm25": "capitale Australie",
   "title_hints": [],
   "persons": [],
   "companies": [],
   "published_after": null,
   "published_before": null,
   "description_requested": false,
   "transcription_requested": false,
   "top_k": 40,
   "final_k": 5
  },
  "vector_search": {
   "prefilter": {
    "sql": null,
    "params": [],
    "applied": false,
    "sql_prefilters": false,
    "candidate_count": null,
    "candidate_chunk_ids": null,
    "general_question_only": true
   }
  },
  "retrieval": {
   "bm25": {
    "result_count": 0
   },
   "vector": {
    "result_count": 40
   },
   "rrf": {
    "rank_constant": 60,
    "bm25_count": 0,
    "vector_count": 40,
    "fused_count": 30
   },
   "rerank": {
    "applied": true,
    "input_count": 30,
    "output_count": 5,
    "relevance_scores": [
     0.13734724,
     0.1103177,
     0.10879345,
     "… +2 autres"
    ]
   },
   "passages": [
    {
     "video_title": "Rendez-vous de la double compétence : « COP 21 : et après ? Vers une révolution énergétique »",
     "text": "Ces hydrocarbures non-conventionnels peuvent, dans le futur, prendre une place grandissante, ce sera sans doute le cas, parce qu'aujourd'hui, c'est simplement l'Amérique du Nord, les États-Unis, le Canada, mais il y a des réserves importantes, des ressources importantes, qui sero…"
    },
    {
     "video_title": "Julien Marchal, Directeur Stratégie et Développement Durable, EDF Renouvelables",
     "text": "Globalement, on analyse l'impact de nos parcs sur l'environnement au sens large, à la fois la biodiversité, l'empreinte carbone, l'artificialisation des sols et toute une autre série de facteurs. Et on va chercher à optimiser nos parcs sur ces différents critères et à regarder le…"
    },
    {
     "video_title": "Discours de Joël de Rosnay, prospectiviste, parrain de la promotion 2016",
     "text": "La première des choses que je voudrais vous dire, c'est que mon message pour vous est un message d'espoir vers l'avenir. Je voudrais que vous soyez constamment conscients qu'on peut construire positivement notre avenir. Pour ça, j'ai quatre mots, c'est comprendre, vouloir... aime…"
    },
    "… +2 autres"
   ]
  },
  "generate": {
   "answer": "Je n'ai pas trouvé dans les vidéos d'élément permettant de répondre à cette question.",
   "action": "abstain",
   "source_indexes": []
  },
  "persist": {
   "conversation_id": 816,
   "message_id": 923,
   "memory": {
    "current_topic": "capitale de l'Australie",
    "follow_up": false,
    "topic_changed": false,
    "videos_discussed": []
   }
  },
  "end": {
   "action": "abstain",
   "answer": "À première vue, je n'ai pas la réponse dans les vidéos.",
   "sources": []
  }
 }
};
