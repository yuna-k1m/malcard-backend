from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import health, cards, analysis

app = FastAPI(title="MalCard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(cards.router, prefix="/cards", tags=["cards"])
app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])