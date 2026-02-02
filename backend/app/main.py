from fastapi import FastAPI

from app.api import forecast, adversarial, inventory, rebalancing, auth, analytics, products, users, purchase_orders, llm_chat, csv_upload, simulations, gnn_graph
from fastapi.middleware.cors import CORSMiddleware

# Try to import gnn, but don't fail if torch is not available
try:
    from app.api import gnn
    gnn_available = True
except ImportError:
    gnn_available = False

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
app.include_router(analytics.router)
app.include_router(products.router)
app.include_router(users.router)
app.include_router(purchase_orders.router)
app.include_router(llm_chat.router)
app.include_router(csv_upload.router)
app.include_router(simulations.router)
app.include_router(gnn_graph.router)

if gnn_available:
    app.include_router(gnn.router)


