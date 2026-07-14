# Car Deal Radar — Roadmap détaillée

Plan de travail pour terminer le projet, phase par phase, en suivant les
semaines 6, 7 et 8 du cours officiel
([ed-donner/llm_engineering](https://github.com/ed-donner/llm_engineering)).
Chaque étape indique le fichier du cours qui sert de référence, les fichiers de
ce projet à modifier, et un critère de fin (« Done quand »).

**État actuel** : squelette terminé — modèles de données, baseline hors-ligne,
évaluation, `DealFinder`, démo et 10 tests. `FrontierLLMPricer.predict()` et
`FineTunedPricer.predict()` sont des placeholders.

---

## Phase 1 — Prédiction de prix avec des modèles frontier (semaine 6)

### 1.1 Obtenir un vrai jeu de données d'annonces

- Choisir une source d'annonces auto israéliennes (export manuel, dataset
  public, ou scraping — vérifier les CGU du site avant tout scraping).
- Convertir les données brutes au format JSON de `data/sample_cars.json`
  (mêmes champs que `CarListing`).
- Écrire la conversion dans un petit script séparé (hors package) ou étendre
  `load_car_listings` si le format d'entrée est proche.
- Référence cours : `week6/pricer/loaders.py` + `parser.py` (chargement +
  parsing par datapoint, retour `None` pour les rejets).
- **Done quand** : ≥ quelques centaines d'annonces réelles chargées et validées
  par `validate_listing`.

### 1.2 Curation et analyse du dataset

- Filtrer les annonces invalides (utiliser les messages de `validate_listing`).
- Analyser les distributions (prix, année, kilométrage, marques) dans un
  notebook, comme `week6/day1.ipynb`/`day2.ipynb` le fait pour les produits :
  histogrammes, valeurs aberrantes, tranches de prix retenues.
- Décider des bornes de prix min/max à garder (le cours élague les extrêmes).
- **Drift temporel** : si l'entraînement se fait sur un dataset historique
  (ex. Kaggle, arrêté mi-2024) et l'inférence sur des annonces fraîchement
  scrapées, mesurer l'écart : garder un petit échantillon d'annonces récentes
  comme jeu de calibration, comparer l'erreur dessus vs le test set
  historique, et si besoin appliquer un facteur de correction global
  (régression prédit → réel sur l'échantillon récent). Réévaluer ce facteur à
  chaque nouvelle collecte.
- Faire le split avec `split_listings` (seed fixe) et le sauvegarder pour que
  train/test restent stables entre expériences.
- Optionnel (fidèle au cours) : pousser le dataset sur le Hugging Face Hub
  comme `Item.push_to_hub` (`week6/pricer/items.py`) — nécessite `HF_TOKEN`.
- **Done quand** : un train/test figé et documenté (tailles, bornes de prix).

### 1.3 Baselines traditionnelles (optionnel mais recommandé)

- Le cours (`week6/day3.ipynb`) compare des baselines scikit-learn (features +
  LinearRegression, RandomForest, XGBoost) avant les LLM.
- Adapter : features numériques année/kilométrage/carburant → régression
  linéaire, en plus du `BaselinePricer` actuel.
- Ajouter chaque baseline comme sous-classe de `BasePricer` dans `pricers.py`.
- Dépendance : `scikit-learn>=1.7.2` (présente dans le pyproject officiel) —
  l'ajouter seulement à ce moment-là.
- **Done quand** : au moins une baseline ML évaluée sur le test set.

### 1.4 Implémenter `FrontierLLMPricer.predict()`

- Installer le groupe `frontier` : `uv sync --extra frontier`.
- Suivre `week6/day4.ipynb` : appels via `litellm.completion(model=...,
  messages=[...], seed=42)` — PAS de LangChain.
- Charger les clés avec `load_dotenv(override=True)` (déjà en place dans
  `main.py`) ; `.env` à partir de `.env.example`.
- Utiliser `build_prompt()` (déjà écrit et testé) et `parse_response()` pour
  transformer la réponse JSON en `PricePrediction`.
- Prévoir un retry simple si le JSON est invalide (1 relance max, puis erreur).
- Tester avec 2-3 modèles : un modèle OpenAI, un Claude, éventuellement un
  Gemini (mêmes familles que `week6/day4.ipynb`).
- Ajouter des tests avec des réponses simulées (pas d'appel réseau en test).
- **Done quand** : `evaluate_pricer(FrontierLLMPricer(...), test)` produit des
  métriques réelles sur le test set.

### 1.5 Benchmark et visualisation

- Comparer baseline(s) vs modèles frontier avec `calculate_metrics`.
- Reproduire les graphiques du cours (`week6/pricer/evaluator.py`) : scatter
  prix réel vs prix prédit avec code couleur vert/orange/rouge, et courbe
  d'erreur moyenne cumulée. Dépendances cours : `plotly`, `pandas`,
  `matplotlib` (à ajouter à ce moment-là).
- Ajouter des découpages spécifiques auto : erreur par marque, par année, par
  tranche de prix.
- Faire ce travail dans un notebook (comme le cours) plutôt que dans le
  package.
- **Done quand** : un tableau comparatif des modèles + graphiques, committés
  dans un notebook `notebooks/phase1_benchmark.ipynb`.

---

## Phase 2 — Fine-tuning QLoRA d'un modèle open source (semaine 7)

### 2.1 Construire le dataset de fine-tuning

- Partir de `create_fine_tuning_records` (déjà écrit, format
  prompt/completion identique à `week7/pricer/items.py` :
  `"{QUESTION}\n\n{texte}\n\n{PREFIX}"` → `"{prix arrondi}.00"`).
- Ajouter la troncature par tokens comme `Item.make_prompts(tokenizer,
  max_tokens)` du cours : couper la description pour tenir dans le budget de
  tokens du modèle choisi.
- Convertir en `datasets.Dataset` et pousser sur le Hub au format
  train/val/test comme `Item.push_prompts_to_hub`.
- Dépendances : groupe `finetuning` (`datasets`, `transformers`, `torch`).
- **Done quand** : dataset prompt/completion sur le Hub (ou en local),
  vérifié à la main sur quelques exemples.

### 2.2 Entraîner avec QLoRA

- Suivre le notebook Colab lié depuis `week7/day3 and 4.ipynb` (l'entraînement
  du cours se fait sur Colab avec GPU, pas dans le repo).
- Stack du cours : `transformers` + `bitsandbytes` (quantization 4-bit) +
  `peft` (LoRA) + `trl` (`SFTTrainer`), suivi éventuel avec `wandb`.
- Choisir le même modèle de base que le cours (vérifier dans le Colab au
  moment de commencer — le cours a utilisé un modèle 8B type Llama).
- Hyperparamètres : partir de ceux du Colab du cours, ne changer qu'une chose
  à la fois.
- Sauvegarder les adapters LoRA sur le Hugging Face Hub.
- **Done quand** : adapters entraînés et poussés sur le Hub, courbe de loss
  saine.

### 2.3 Implémenter `FineTunedPricer.predict()`

- Charger le modèle de base + adapters (`peft`), générer la suite du prompt
  `"Price is ₪"` et parser le nombre produit (comme le cours parse
  `"Price is $"` — voir `Tester.post_process` dans
  `week6/pricer/evaluator.py` : regex extrayant le premier nombre).
- Prévoir le chargement paresseux (ne charger le modèle qu'au premier
  `predict`) pour que l'import du module reste léger.
- Ajouter des tests avec un générateur simulé (pas de GPU en test).
- **Done quand** : `FineTunedPricer` évalué sur le test set.

### 2.4 Comparaison finale frontier vs fine-tuné

- Rejouer le benchmark 1.5 avec le modèle fine-tuné inclus.
- Reproduire la conclusion de la semaine 7 : le spécialiste fine-tuné
  peut-il battre les modèles frontier sur ce domaine ?
- **Done quand** : tableau final des métriques committé, avec analyse courte.

---

## Phase 3 — Système autonome de détection de deals (semaine 8)

### 3.1 Scanner d'annonces (`ListingScanner`)

- Extraire de `DealFinder` la partie « acquisition » : récupérer les nouvelles
  annonces et ignorer celles déjà vues.
- Référence : `week8/agents/scanner_agent.py` (mémoire d'URLs déjà traitées)
  et `deal_agent_framework.py` (`memory.json` lu/écrit en JSON simple).
- Implémenter la même mémoire : fichier `memory.json` avec les
  `listing_id`/URLs déjà signalés.
- **Done quand** : deux exécutions successives ne re-signalent pas les mêmes
  annonces.

### 3.2 RAG sur le prédicteur frontier

- C'est ici que le RAG entre dans le projet du cours : dans
  `week8/agents/frontier_agent.py`, le FrontierAgent ne demande pas un prix
  « à froid » — il cherche les 5 produits les plus similaires dans un
  vectorstore ChromaDB (embeddings `sentence-transformers/all-MiniLM-L6-v2`)
  et injecte leurs descriptions + prix dans le prompt comme contexte.
- Construire le vectorstore à partir du train set (comme `week8/day2.ipynb`
  remplit `products_vectorstore`) : une entrée par annonce, document =
  `to_model_text()`, métadonnée = prix.
- Étendre `FrontierLLMPricer` (ou créer une variante RAG) : recherche des 5
  voitures similaires → contexte « Potentially related car: ... Price is
  ₪... » → appel frontier.
- Dépendances à ajouter à ce moment-là : `chromadb>=1.1.0`,
  `sentence-transformers>=5.1.1` (versions du pyproject officiel).
- Mesurer l'apport : erreur avec RAG vs sans RAG sur le test set.
- **Done quand** : le pricer frontier + RAG bat le pricer frontier seul.

### 3.3 Ensemble de modèles

- Référence : `week8/agents/ensemble_agent.py` — combinaison pondérée des
  prédicteurs (le cours pondère frontier 0.8 / spécialiste 0.1 / réseau de
  neurones 0.1).
- Remplacer la moyenne simple de `DealFinder.estimate_price` par une
  pondération apprise ou fixée à partir des résultats de la phase 2.
- **Done quand** : l'estimation agrégée bat chaque prédicteur individuel sur
  le test set (ou la déviation est documentée).

### 3.4 Notifications Pushover

- Référence : `week8/agents/messaging_agent.py` — `requests.post` sur
  `https://api.pushover.net/1/messages.json` avec `PUSHOVER_USER` /
  `PUSHOVER_TOKEN` (déjà dans `.env.example`).
- Créer un composant de notification qui envoie le meilleur `DealCandidate`
  (prix, estimation, remise, URL), seulement au-dessus d'un seuil de score.
- Optionnel : rédaction du message par un LLM comme `craft_message()` du cours.
- Dépendance : groupe `notifications` (`requests`).
- Tester avec un mock de `requests.post` (pas d'envoi réel en test).
- **Done quand** : une vraie notification reçue sur téléphone pour un deal
  au-dessus du seuil.

### 3.5 Orchestration planifiée

- Référence : `week8/agents/planning_agent.py` (workflow scan → estimation →
  tri par remise → alerte si remise > seuil) et `deal_agent_framework.py`
  (boucle + mémoire persistante).
- Découper `DealFinder` en : scanner (3.1), estimateur/ensemble (3.2-3.3),
  planificateur (ce point), messagerie (3.4). Garder les mêmes signatures
  publiques tant que possible pour ne pas casser les tests.
- Ajouter un point d'entrée `python -m car_deal_radar.radar` (ou étendre
  `main.py`) qui exécute un cycle complet.
- Pour la planification récurrente, commencer par un simple cron local.
- **Done quand** : un cycle complet tourne sans intervention et alimente
  `memory.json`.

### 3.6 Interface et déploiement (optionnel, comme la fin du cours)

- Gradio : le cours termine avec `week8/price_is_right.py` (UI Gradio montrant
  les deals et les logs). Ajouter une petite UI Gradio listant les
  `DealCandidate` triés. Dépendance : `gradio>=5.47.2,<6.0`.
- Modal : le cours déploie le modèle fine-tuné comme service distant
  (`week8/pricer_service2.py`, classe `Pricer` appelée via
  `modal.Cls.from_name` dans `specialist_agent.py`). À faire seulement si tu
  veux exécuter l'inférence hors de ta machine. Dépendance : `modal>=1.1.4`.
- **Done quand** : (si retenu) UI locale qui affiche les deals du dernier run.

---

## Transversal (au fil des phases)

- **Tests** : chaque nouvelle fonctionnalité arrive avec 1-3 tests rapides,
  hors-ligne, avec mocks pour API/GPU. Garder `pytest` vert en permanence.
- **Dépendances** : n'installer un groupe (`frontier`, `finetuning`,
  `notifications`) qu'au début de la phase qui l'utilise ; toute nouvelle
  dépendance doit exister dans le pyproject officiel du cours (sinon la
  documenter comme déviation dans le README).
- **Secrets** : tout passe par `.env` (jamais committé) ; `.env.example` liste
  les noms de variables.
- **Notebooks** : le travail exploratoire (analyse, benchmarks, courbes) va
  dans `notebooks/`, le code réutilisable dans `src/car_deal_radar/`.
- **README** : mettre à jour la section « Course alignment » à la fin de
  chaque phase.

## Ordre recommandé

1.1 → 1.2 → 1.4 → 1.5 (1.3 en parallèle si souhaité), puis 2.1 → 2.2 → 2.3 →
2.4, puis 3.1 → 3.2 → 3.3 → 3.4 → 3.5 (→ 3.6). Chaque phase se termine par la
mise à jour du benchmark et du README.
