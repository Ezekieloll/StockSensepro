# StockSense - AI-Powered Inventory Management

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.11+** 
- **Node.js 18+**
- **PostgreSQL 14+**
- **Git**

---

## 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/StockSense.git
cd StockSense
```

---

## 🗄️ Step 2: Set Up PostgreSQL Database

1. Install PostgreSQL if not already installed
2. Create the database:

```bash
# Open PostgreSQL shell
psql -U postgres

# Create database
CREATE DATABASE stocksense;
\q
```

---

## 🔧 Step 3: Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/stocksense" > .env
```

### Import Transaction Data
```bash
# This imports 2023-2024 transaction data into PostgreSQL
python scripts/import_transactions.py

# Generate simulated data from 2024 to today
python scripts/daily_simulator.py --catch-up
```

### Start Backend Server
```bash
uvicorn app.main:app --reload
# Runs on http://localhost:8000
```

---

## 🤖 Step 4: ML Service Setup

```bash
cd ../ml

# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or if using PyTorch:
pip install torch numpy pandas fastapi uvicorn

# Start ML inference service
uvicorn inference_api:app --reload --port 8001
# Runs on http://localhost:8001
```

---

## 🎨 Step 5: Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
# Runs on http://localhost:3000
```

---

## ✅ Step 6: Verify Everything is Running

Open 3 terminals and run:

| Terminal | Directory | Command |
|----------|-----------|---------|
| 1 | `backend/` | `uvicorn app.main:app --reload` |
| 2 | `ml/` | `uvicorn inference_api:app --reload --port 8001` |
| 3 | `frontend/` | `npm run dev` |

### URLs:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000/docs
- **ML Service**: http://localhost:8001/health

---

## 🔐 Default Login Credentials

| Email | Password | Role |
|-------|----------|------|
| admin@stocksense.com | admin123 | Admin |
| manager@stocksense.com | manager123 | Manager |

---

## 📊 Project Structure

```
StockSense/
├── backend/           # FastAPI backend
│   ├── app/
│   │   ├── api/       # API endpoints
│   │   ├── models/    # Database models
│   │   └── services/  # Business logic
│   ├── scripts/       # Data import scripts
│   └── .env           # Environment variables
│
├── frontend/          # Next.js frontend
│   ├── app/           # Pages
│   ├── components/    # Reusable components
│   └── .env.local     # Frontend config
│
└── ml/                # ML/AI models
    ├── forecasting/   # TFT+GNN model
    ├── data/          # Transaction data
    └── inference_api.py  # ML service
```

---

## 🛠️ Troubleshooting

### Database Connection Error
- Make sure PostgreSQL is running
- Check `.env` file has correct DATABASE_URL
- Verify database `stocksense` exists

### Frontend Can't Connect to Backend
- Check backend is running on port 8000
- Verify CORS is enabled in backend
- Check `.env.local` has correct API URL

### ML Service Not Loading Model
- Ensure PyTorch is installed
- Check model file exists: `ml/models/best_tft_gnn_v2.pt`

---

## 📝 Commands Summary

```bash
# Terminal 1 - Backend
cd backend
.venv\Scripts\activate
python scripts/import_transactions.py  # First time only
python scripts/daily_simulator.py --catch-up  # Catch up to today
uvicorn app.main:app --reload

# Terminal 2 - ML Service
cd ml
venv\Scripts\activate
uvicorn inference_api:app --reload --port 8001

# Terminal 3 - Frontend
cd frontend
npm run dev
```

---

## 🎉 You're Ready!

Visit http://localhost:3000 and login with the credentials above.
