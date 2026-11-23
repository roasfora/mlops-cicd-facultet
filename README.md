# mlops-cicd-facultet

**Class 06 — Practice: Set up CI/CD for ML Models** (Facultét)

A small, executable MLOps teaching project. It predicts Titanic passenger survival
(binary classification) and shows how **Continuous Integration** protects an ML project:
every push trains the model and runs data tests, model-quality tests and API tests on GitHub Actions.

> **Scope:** this project implements **CI** plus a **local** API demonstration.
> It does **not** deploy anything automatically to production.

Stack: Python 3.11 · scikit-learn · pandas · pytest · joblib · MLflow · FastAPI · GitHub Actions

---

## 1. Project layout

```
.github/workflows/ci.yml   CI pipeline (GitHub Actions)
data/train.csv, test.csv   80/20 split of the Titanic data (random_state=42)
src/prepare_data.py        downloads Titanic and writes the split to data/
src/train.py               trains the pipeline, logs to MLflow, saves model.joblib
src/app.py                 FastAPI service: GET /health, POST /predict
tests/test_data.py         data tests      (is the DATA valid?)
tests/test_model.py        model tests     (is the MODEL good enough?)
tests/test_api.py          API tests       (does the SERVICE work?)
Makefile                   install / train / test / ci / serve
```

Dataset source: the public
[seaborn-data Titanic CSV](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv).
Features: `pclass`, `sex`, `age`, `fare`. Target: `survived`.
The test set is used **only** for evaluation — never for fitting the imputer, encoder or model.

## 2. Quick start

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
make install
make train                       # trains, prints metrics, writes model.joblib
make test                        # runs all pytest tests
make ci                          # = train + test (what GitHub Actions does)
make serve                       # API at http://127.0.0.1:8000/docs
```

`data/train.csv` and `data/test.csv` are committed. To regenerate them: `python src/prepare_data.py`.

### Windows equivalents (no `make` needed)

| make command   | Windows (PowerShell, venv activated)                              |
|----------------|-------------------------------------------------------------------|
| `make install` | `pip install -r requirements.txt`                                 |
| `make train`   | `python src/train.py`                                             |
| `make test`    | `python -m pytest -v`                                             |
| `make ci`      | `python src/train.py; python -m pytest -v`                        |
| `make serve`   | `python -m uvicorn src.app:app --reload`                          |

### Try the API

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"pclass": 1, "sex": "female", "age": 29, "fare": 80}'
```

## 3. MLflow experiment tracking

`src/train.py` logs to the experiment **`facultet-mlops-cicd`**: hyperparameters, `accuracy`,
`f1`, and the trained pipeline. Tracking is fully local (a `mlflow.db` SQLite file — git-ignored).

```bash
make train                                   # run it a few times (change PARAMS in train.py!)
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000 → experiment "facultet-mlops-cicd" → compare runs
```

Classroom idea: change `max_depth` or `n_estimators` in `PARAMS`, retrain, and compare runs in the UI.

---

## 4. Live demo script

> Teach with **branches and pull requests** — don't break `main` on purpose.
> Prerequisite: the repo is pushed and the venv is active.

### DEMO A — Green pipeline

```bash
git switch -c teaching-demo
make ci                       # all 13 tests pass locally
git push -u origin teaching-demo
```
Open a pull request into `main` → **Actions** tab → **CI** workflow.
Show every step (checkout, install, train, 3 test steps, artifact upload) turning green,
and download the `model` artifact from the run summary.

### DEMO B — Break the data

```bash
python -c "import pandas as pd; d=pd.read_csv('data/train.csv').drop(columns=['fare']); d.to_csv('data/train.csv', index=False)"
make test                     # (not `make ci`: training itself would crash on the missing column)
```
Expected: `test_required_columns_exist` fails with `Missing columns: ['fare']`
and `test_feature_data_types_are_valid` fails with `KeyError: 'fare'`.

**Explain:** nothing in the *code* changed, yet the project is broken. A data schema test
catches this before a model is trained on bad data or served to users.

### DEMO C — Break the model quality check

First restore the data: `git restore data/train.csv`. Then in `tests/test_model.py` set:

```python
MIN_ACCURACY = 1.01   # impossible: accuracy can never exceed 1.0
```
```bash
make test
```
Expected: `test_accuracy_meets_minimum_threshold` fails with `accuracy 0.832 < 1.01`.

**Explain:** this is a *quality gate*. In real life a drop below the threshold means a bad
model must not move forward. (The real threshold, `0.75`, is an educational example.)

### DEMO D — Fix the project

```bash
git restore data/train.csv tests/test_model.py   # restore dataset + threshold (0.75)
make ci                                          # green again
git status                                       # nothing left to commit if you never committed the breakage
```
If you pushed the broken state (recommended, so students see a red ✗ on the PR), commit the fix and push:
```bash
git commit -am "Fix: restore dataset and accuracy threshold"
git push
```
Show the new workflow run turning green and the PR becoming mergeable.

---

## 5. MLOps concepts

**What is MLOps?** Practices that bring DevOps ideas (automation, testing, versioning, monitoring)
to machine learning, so models can be built, validated and delivered reliably and repeatably.

**What is Continuous Integration (CI)?** Every change is automatically built and tested
(here: train the model, run data/model/API tests) so problems are found within minutes.

**What is Continuous Delivery (CD)?** Every change that passes CI is automatically *packaged and
made ready to release* (Continuous Deployment goes one step further and releases it automatically).

**CI vs CD.** CI answers *"is this change safe to merge?"*. CD answers *"how does a verified
change reach users?"*. This project only does CI; it uploads the model as a build artifact but deploys nothing.

**Why data tests, not only code tests?** In ML the data is part of the program. A missing
column, new category or corrupted labels can break a model without a single line of code changing.

**Why model performance tests?** Code can run perfectly and still produce a useless model.
A minimum-accuracy gate turns "the model got worse" into a failing build.

**Role of MLflow.** Records each training run — parameters, metrics, the model — so experiments
are comparable and reproducible instead of lost in notebooks.

**Role of GitHub Actions.** Runs the pipeline on a clean machine on every push and pull
request. "Works on my laptop" is no longer enough, and nobody has to remember to run the tests.

**This project vs Azure ML / AWS SageMaker.** Those are managed platforms: scalable training
clusters, model registries, managed endpoints, pipelines, monitoring, and cost/IAM management.
This project is the small, free, transparent core of the same idea — automated checks around
train → test — without cloud infrastructure.

### Architecture

```
 Developer ──git push / PR──▶ GitHub ──▶ GitHub Actions (ubuntu, Python 3.11)
                                              │
                      ┌───────────────────────┼─────────────────────────┐
                      ▼                       ▼                         ▼
              install requirements      python src/train.py      pytest (3 stages)
                                        ├─ model.joblib          ├─ data tests
                                        └─ MLflow (local)        ├─ model tests (accuracy ≥ 0.75)
                                                                 └─ API tests
                                              │
                          all green? ──yes──▶ upload model.joblib as artifact
                                     └─no───▶ workflow fails ✗ (PR shows red)

 Local only:  make serve ─▶ FastAPI (/health, /predict) loads model.joblib
```

## 6. Notes

- `model.joblib`, `mlflow.db`, `mlruns/` and `.env` files are git-ignored and never committed.
- Models are logged to MLflow in `cloudpickle` format because we train and trust them ourselves.
  Never load pickled models from untrusted sources.
- The CI workflow has read-only permissions (`contents: read`) and needs no secrets.
