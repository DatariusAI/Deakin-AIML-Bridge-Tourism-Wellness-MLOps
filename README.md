# Wellness Tourism Package: Purchase Prediction MLOps Pipeline

Deakin Applied AI Master Bridge Program · Advanced Machine Learning and MLOps · v1.1

"Visit with Us" wants to know which customers will buy the new Wellness Tourism Package before calling them.
This repo holds an end-to-end pipeline that registers the data, prepares it, tunes an XGBoost classifier with MLflow tracking,
and commits the best model back to `main`. A Streamlit app serves the model.

## Pipeline (GitHub Actions)
`.github/workflows/pipeline.yml` runs on every push to `main`:

1. **register-dataset** → `tourism_project/model_building/data_register.py` validates `tourism_project/data/tourism.csv`
2. **data-prep** → `prep.py` cleans the data and writes an 80/20 stratified split (artifact `data-splits`)
3. **model-traning** → `train.py` runs GridSearchCV (24 combos, 5-fold, F1), logs every combo to MLflow, saves
   `tourism_project/deployment/best_tourism_model_v1.joblib` and commits it with `[skip ci]`

## Results (v1, held-out test set of 803 customers)
| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.940 | 0.864 | 0.819 | 0.841 | 0.961 |

## Live apps
- Streamlit Community Cloud: https://datariusai-tourism-wellness-mlops.streamlit.app
- Hugging Face Space: https://huggingface.co/spaces/DatariusAI/tourism-wellness-mlops (Docker; `hf_space/Dockerfile`)

`.github/workflows/keepalive.yml` opens both apps every 6 hours so they do not sleep.

Locally: `pip install -r tourism_project/deployment/requirements.txt && streamlit run tourism_project/deployment/app.py`

## Notebook
`notebooks/Tourism_Wellness_MLOps_Notebook_v1.1.ipynb` (presentation in `docs/`) contains every script plus the executed local run.
