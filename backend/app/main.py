from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api import forecast, adversarial, inventory, rebalancing, auth, analytics, products, users, purchase_orders, llm_chat, csv_upload, simulations, gnn_graph, audit_logs
from fastapi.middleware.cors import CORSMiddleware
from app.config import ALLOWED_ORIGINS, ALLOWED_ORIGIN_REGEX
from app.database import get_db
from app.services.auth_utils import oauth2_scheme, decode_access_token

# Try to import gnn, but don't fail if torch is not available
try:
    from app.api import gnn
    gnn_available = True
except ImportError:
    gnn_available = False

app = FastAPI(title="StockSense Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)


def require_auth(token: str = Depends(oauth2_scheme)) -> dict:
    return decode_access_token(token)


secured = [Depends(require_auth)]


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "db": "connected"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

app.include_router(forecast.router, dependencies=secured)
app.include_router(adversarial.router, dependencies=secured)
app.include_router(inventory.router, dependencies=secured)
app.include_router(rebalancing.router, dependencies=secured)
app.include_router(analytics.router, dependencies=secured)
app.include_router(products.router, dependencies=secured)
app.include_router(users.router, dependencies=secured)
app.include_router(purchase_orders.router, dependencies=secured)
app.include_router(llm_chat.router, dependencies=secured)
app.include_router(csv_upload.router, dependencies=secured)
app.include_router(simulations.router, dependencies=secured)
app.include_router(gnn_graph.router, dependencies=secured)
app.include_router(audit_logs.router, dependencies=secured)

if gnn_available:
    app.include_router(gnn.router, dependencies=secured)


