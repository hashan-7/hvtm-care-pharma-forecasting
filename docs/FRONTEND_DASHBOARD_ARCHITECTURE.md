# HVTM CARE - Frontend Web Dashboard Architecture

## 1. Executive Summary & UI/UX Objectives
The **HVTM CARE Web Dashboard** is a responsive, single-page client interface designed for pharmaceutical inventory managers and healthcare procurement officers. It bridges the gap between complex machine learning forecasting models (XGBoost, NeuroStack, GRU, BiLSTM) and clinical inventory workflows.

The dashboard delivers real-time stock-out warnings, dynamic generic-to-brand mapping, automated reorder recommendations, and seamless RESTful integration with the FastAPI backend engine.

---

## 2. Interface Design System & Layout Architecture

### Design Philosophy
- **Modern Glassmorphic Aesthetic:** Utilizes subtle background blurs (`backdrop-filter: blur(10px)`), soft neutral palettes (Slate-50 to Slate-900), and crisp borders (`border-slate-100/200`) to provide an uncluttered, hospital-grade operational view.
- **Micro-Interactions & Responsive Layout:** Powered by Tailwind CSS utilities and Lucide icons, ensuring adaptability across mobile devices, tablets, and full-resolution pharmacy desktop terminals.

### Core Views & Navigation
1. **Security Gateway (`#login-screen`):**
   - Session-based authentication with auto-expiration (24-hour timeout check via `localStorage`).
   - Secure input sanitization and access status notifications.
2. **Operational Dashboard (`#dashboard-screen`):**
   - **Sidebar Navigation:** Modular tab switching across *Predict Demand*, *Add Inventory Record*, *Batch Processing*, and *Inventory Logs*.
   - **Metrics Overview Bar:** Real-time macroeconomic indicator tracking (`currentUSD` FX rate) and inventory status summaries.
   - **Dynamic Drug Selector:** Dynamic brand dropdown populated instantly based on active generic selection via `DRUG_BRAND_MAP`.
   - **Prediction Results Display:** Visual cards presenting forecast demand units, dynamic safety stock buffer, and days-to-stockout indicators.
   - **Historical Inventory Table:** Tabular log of past entries with real-time record deletion modals and refresh handlers.

---

## 3. Client-Side Modular Architecture

The frontend JavaScript codebase is organized into modular scripts for clean separation of concerns:

| Script File | Primary Responsibility | Technical Details |
| :--- | :--- | :--- |
| **`js/api.js`** | Backend API Service Client | Encapsulates asynchronous `fetch()` calls to `/predict`, `/data/add`, `/data/all`, `/data/delete`, and health check endpoints. |
| **`js/auth.js`** | Client Authentication & Session | Manages user credentials validation, session persistence, and logout teardown. |
| **`js/navigation.js`** | View Routing & State Switching | Controls view swapping, active sidebar tab highlights, and mobile drawer toggling. |
| **`js/ui_handlers.js`** | Interactive Logic & Form Handling | Houses `DRUG_BRAND_MAP`, dynamic select population, form validation, Toast notifications, and result rendering. |

---

## 4. RESTful API Integration & Asynchronous State Flow

The client communicates with the FastAPI service asynchronously:

```
[ User Input / Form Submission ]
               │
               ▼
   [ ui_handlers.js Validation ]
               │
               ▼
   [ api.js Async Fetch Request ] ────► POST /predict (JSON Payload)
               │                                │
               ▼                                ▼
   [ Response Parsing & State ]   ◄──── [ FastAPI Backend / ML Pipeline ]
               │
               ▼
   [ DOM Update & Toast Notification ]
```

- **Payload Construction:** Automatically constructs feature payloads matching the backend's expected 38-feature vector (including generic one-hot vectors, macroeconomic rates, and historical lag attributes).
- **Error Handling & Feedback:** Catches network failures and HTTP exceptions, surfacing non-intrusive UI alerts without freezing the user workflow.

---

## 5. Security & Session Hygiene
- Local storage auth tokens are validated on every route access via `checkSession()`.
- Expired sessions trigger automatic state cleanup and redirection to the security gateway.
