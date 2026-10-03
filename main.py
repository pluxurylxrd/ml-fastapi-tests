from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentiment import predict, load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(
    title="Sentiment API",
    description="Определение тональности отзывов покупателей маркетплейса",
    version="1.0.0",
    lifespan=lifespan,
)

class ReviewRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
        description="Текст отзыва",
        examples=["Заказ пришёл быстро, качество отличное"],
    )

class SentimentResponse(BaseModel):
    label: str
    score: float
    probabilities: dict[str, float]

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: ReviewRequest):
    try:
        return predict(request.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))