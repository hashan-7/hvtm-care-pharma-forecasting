# HVTM CARE - Backend REST API Architecture

## 1. System Overview & Technology Stack
The **HVTM CARE Backend** is an asynchronous, high-throughput RESTful microservice built with **FastAPI** and **Python 3.10+**. It serves as the central operational bridge between the client-facing Web Dashboard (`frontend/`) and the predictive machine learning engine (`backend/engine_logic.py`).

The backend handles request validation, session management, inventory data persistence, stock-out mathematical projections, and model inference orchestration.

---

## 2. API Endpoints Specification

### 2.1 Predictive Inference Service
- **Endpoint:** `POST /predict`
- **Payload Schema (`PredictionInput`):**
  ```json
  {
    "brand_name": "Metfogen-XR",
    "current_stock": 1500.0,
    "usd_rate": 310.50
  }
  ```
- **Operational Flow:**
  1. Validates drug entity existence against `DRUG_STRENGTH_MAP`.
  2. Forwards validated parameters to `run_prediction()` in `engine_logic.py`.
  3. Computes dynamic safety buffer, daily consumption rate, and days-to-stockout.
  4. Returns real-time JSON response with forecast demand and suggested replenishment order volume.

### 2.2 Inventory Data Management & Persistence
- **`GET /data/logs`**: Retrieves historical monthly inventory records from disk, sorted chronologically with null-handling.
- **`POST /data/add`**: Validates input against future-date constraints and duplicate month-brand entries using `check_duplicate_and_lock()`. Appends verified entries to the historical dataset.
- **`POST /data/delete`**: Safely removes specified historical records, recalibrates lag structures, and updates the local storage ledger.
- **`GET /data/health`**: Returns engine status, loaded model artifacts verification, and API latency health checks.

---

## 3. Data Integrity & Validation Architecture

### Pydantic Data Contracts
Strict type enforcement and schema validation using Pydantic `BaseModel` classes:
- **`PredictionInput`**: Validates numeric bounds for `current_stock` and macroeconomic `usd_rate`.
- **`DataEntryInput`**: Enforces required attributes including `brand_name`, `year`, `month`, `actual_issued_qty`, and `received_qty`.

### Duplicate Prevention & Temporal Guardrails
- **`check_duplicate_and_lock()`**: Enforces strict business logic preventing duplicate logs for the same brand within the same billing month.
- **Temporal Consistency**: Blocks forward-dated entries to prevent future data leakage into training lag structures.

---

## 4. CORS & Security Middleware
Configured with FastAPI `CORSMiddleware` to allow asynchronous browser cross-origin requests from the web dashboard:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 5. Model Inference Orchestration
On service initialization (`@app.on_event("startup")`), the service triggers `load_resources()`, pre-loading model weights (XGBoost regressor, Scalers, NeuroStack meta-learners) into memory to ensure sub-50ms inference latency during active pharmacy operations.
