# ml — AML re-score model service

XGBoost model that re-scores rule-generated laundering candidates to cut false
positives. Part of the Money Flow Graph Explorer monorepo.

## Pipeline
1. Backend runs in **collect mode** (`monitor.model.collectTrainingData=true`) and appends
   one row per rule candidate to `data/training_candidates.csv`
   (columns: `ts,label,<features...>`; `label`=1 if the candidate involved a
   ground-truth fraud edge).
2. Train:
   ```bash
   cd ml
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python train.py --data ../data/training_candidates.csv --out .
   ```
   Produces `model.json` + `feature_names.json` and prints holdout PR-AUC / precision-recall
   by threshold (temporal split — trained on earlier timestamps, tested on later).
3. Serve:
   ```bash
   uvicorn serve:app --host 0.0.0.0 --port 8000   # or: docker build -t aml-ml . && docker run -p 8000:8000 -v $PWD:/app aml-ml
   ```
4. Backend runs in **serve mode** (`monitor.model.enabled=true`, `monitor.model.url`,
   `monitor.model.threshold`): for each rule candidate it POSTs the feature map to
   `/predict` and suppresses candidates scored below the threshold.

## API
- `GET /health` → `{status, model_loaded, n_features}`
- `POST /predict` `{ "features": { "<name>": <value>, ... } }` → `{ "probability": <0..1> }`
  (features are reordered by the saved `feature_names.json`; missing → 0. If no model is
  loaded, returns 1.0 so the backend keeps the candidate — fail-open.)
