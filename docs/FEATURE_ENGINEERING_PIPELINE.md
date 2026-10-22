# Feature Engineering & Preprocessing Pipeline Architecture

## 1. Overview & Data Pipeline Scope
The HVTM CARE forecasting framework relies on high-quality feature representations derived from monthly pharmaceutical inventory ledgers spanning January 2020 through January 2026. 

The Feature Engineering Pipeline transforms raw transactional records (Opening Stock, Received Quantity, Issued Quantity, Closing Stock, and Lead Times) into a structured 38-dimensional feature vector suitable for both tabular models (Random Forest, XGBoost, Linear Regression) and sequential deep architectures (GRU, BiLSTM).

---

## 2. Temporal Standardization & Date Alignment

### 2.1 Chronological Continuity (`scripts/fix_dates.py`)
- Standardizes inconsistent date formats across historical pharmacy records into uniform ISO `YYYY-MM-01` monthly frequency.
- Guarantees 73 contiguous monthly timestamps across all 30 unique pharmaceutical brands without temporal gaps or irregular intervals.

### 2.2 Temporal Train-Test Partitioning
To prevent lookahead bias and temporal data leakage:
- **Training Set:** Historical data from January 2020 through December 2024.
- **Testing Set:** Unseen validation window from January 2025 through January 2026.
- Shuffling is strictly disabled to preserve sequential autocorrelation.

---

## 3. Feature Generation Mechanics

### 3.1 Autoregressive Lag Features
Past demand is the strongest indicator of future pharmaceutical requirements. Twelve autoregressive lag features are engineered per brand:
$$\text{Lag}_k(t) = y(t - k) \quad \text{for } k \in \{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\}$$
- **Short-term dynamics (Lags 1–3):** Capture immediate replenishment shocks and post-shortage rebounds.
- **Medium-term dynamics (Lags 4–6):** Capture quarterly procurement cycles.
- **Annual seasonality (Lag 12):** Directly accounts for 12-month cyclical illness patterns and government tender schedules.

### 3.2 Rolling Window Aggregations
To smooth out high-frequency noise and extract medium-term trends:
- **Rolling Mean (3-Month & 6-Month):**
  $$\mu_w(t) = \frac{1}{w} \sum_{i=1}^{w} y(t - i)$$
- **Rolling Standard Deviation (3-Month):**
  $$\sigma_3(t) = \sqrt{\frac{1}{3} \sum_{i=1}^{3} (y(t - i) - \mu_3(t))^2}$$
  Quantifies demand volatility and supply vulnerability.

### 3.3 Macroeconomic Covariates (`USD_Rate`)
- Sri Lankan Rupee (LKR) to US Dollar (USD) currency exchange rates are mapped across the monthly timeline.
- Reflects the severe foreign exchange currency crisis, enabling models to anticipate import restrictions and price elasticity impacts on drug consumption.

### 3.4 Categorical Encoding (`scripts/check_brand_codes.py`)
- **Brand Encoding:** 30 commercial brand entities encoded using consistent Scikit-Learn `LabelEncoder` (`models/label_encoder.pkl`).
- **Generic Categorization:** 10 fundamental therapeutic classes one-hot encoded to enable cross-brand knowledge transfer.

---

## 4. Normalization & Scaler Transformation

### MinMax & Standard Normalization
Features exhibit vastly differing numerical ranges (e.g., issued quantities exceeding 10,000 units vs USD exchange rate ~300 vs binary flags 0/1).

1. **Input Features Scaler (`models/scaler_X.pkl`):**
   $$X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$
   Fitted strictly on training records to prevent distribution leakage into test sets.
2. **Target Label Scaler (`models/scaler_y.pkl`):**
   $$y_{\text{scaled}} = \frac{y - y_{\min}}{y_{\max} - y_{\min}}$$
   Enables stable loss gradient propagation during neural network backpropagation (GRU, BiLSTM, NeuroStack).
3. **Artifact Serialization:**
   Preprocessed arrays are serialized into numpy matrices (`X_train_scaled.npy`, `y_train_scaled.npy`, `X_test_scaled.npy`, `y_test_scaled.npy`) for reproducible model training.
