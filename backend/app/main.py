from fastapi import FastAPI

from app.api import forecast, adversarial, inventory, rebalancing, auth
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="StockSense Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(forecast.router)
app.include_router(adversarial.router)
app.include_router(inventory.router)
app.include_router(rebalancing.router)
app.include_router(auth.router)


