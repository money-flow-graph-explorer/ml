# Money Flow Graph Explorer — ML

룰 기반으로 탐지된 자금세탁 의심 후보를 재점수화해 오탐(false positive)을 줄이는 XGBoost 모델
서비스. 모노레포의 backend가 실시간으로 호출하는 재점수화(re-score) 게이트다.

## 파이프라인

1. **학습 데이터 수집** — backend를 collect 모드(`monitor.model.collectTrainingData=true`)로
   돌리면, 룰이 발화한 모든 후보(TP/FP 모두)가 피처와 함께 `data/training_candidates.csv`에 쌓인다
   (컬럼: `ts,label,<13개 피처>`; `label=1`은 실제 사기 거래가 연루된 경우).

2. **학습**
   ```bash
   cd ml
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python train.py --data ../data/training_candidates.csv --out .
   ```
   `model.json` + `feature_names.json`을 생성하고, 시간 기준 분할(과거 데이터로 학습, 이후 데이터로
   검증 — 미래 정보 유출 방지)로 holdout PR-AUC와 threshold별 precision/recall을 출력한다.

3. **서빙**
   ```bash
   uvicorn serve:app --host 0.0.0.0 --port 8000
   # 또는: docker build -t aml-ml . && docker run -p 8000:8000 -v $PWD:/app aml-ml
   ```

4. **실시간 게이팅** — backend를 serve 모드(`monitor.model.enabled=true`,
   `monitor.model.url`, `monitor.model.threshold`)로 돌리면, 룰 발화 후보마다 피처를 `/predict`로
   전송하고 threshold 미만 점수는 알럿을 억제한다.

## API

- `GET /health` → `{status, model_loaded, n_features}`
- `POST /predict` `{"features": {"<name>": <value>, ...}}` → `{"probability": <0~1>}`
  피처는 저장된 `feature_names.json` 순서로 재정렬되며, 누락된 값은 0으로 채운다. 모델이 로드되지
  않은 경우 1.0을 반환해(fail-open) backend가 해당 후보를 억제하지 않고 유지하도록 한다.

## 설계 메모

- **fail-open**: ML 서비스 장애나 타임아웃이 곧 탐지 누락으로 이어지지 않도록, 오류 시 항상
  "의심스럽다(1.0)"로 응답한다.
- **라벨 유출 방지**: 피처 집합에는 `isFraud`, `alertId` 같은 정답 라벨을 직접 노출하는 필드를
  포함하지 않는다.
