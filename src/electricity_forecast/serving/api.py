"""FastAPI app with /health and /predict endpoints."""


from fastapi import FastAPI, HTTPException

from electricity_forecast.serving.predict import PredictRequest, PredictResponse, load_predictor

app = FastAPI(
    title="Electricity Price Forecast API",
    version="0.1.0",
    description="Day-ahead electricity price forecasting",
)

_predictor = None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = load_predictor()
    return _predictor


@app.get("/health")
def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Return point forecast and optional prediction intervals.
    Accepts timestamp(s), optional weather inputs.
    """
    try:
        pred = _get_predictor()
        return pred.predict(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
