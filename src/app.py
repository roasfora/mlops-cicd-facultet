"""Small FastAPI service to serve the trained model (local demo only).

Run:  uvicorn src.app:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
models = {}  # holds the pipeline once loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Code before `yield` runs once, when the app starts."""
    if MODEL_PATH.exists():
        models["pipeline"] = joblib.load(MODEL_PATH)
    yield
    models.clear()


app = FastAPI(title="Titanic survival API", lifespan=lifespan)


class Passenger(BaseModel):
    """Pydantic rejects invalid input automatically (HTTP 422)."""

    pclass: int = Field(ge=1, le=3, description="Ticket class: 1, 2 or 3")
    sex: Literal["male", "female"]
    age: float | None = Field(default=None, ge=0, le=120)  # missing age is allowed
    fare: float = Field(ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "pipeline" in models}


@app.post("/predict")
def predict(passenger: Passenger):
    if "pipeline" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `make train`.")
    row = pd.DataFrame([passenger.model_dump()])
    survived = int(models["pipeline"].predict(row)[0])
    return {"survived": survived, "label": "survived" if survived else "did not survive"}
