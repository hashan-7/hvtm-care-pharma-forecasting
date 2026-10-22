# Bidirectional LSTM (BiLSTM) Deep Learning Architecture

## 1. Overview & Theoretical Motivation
While standard unidirectional recurrent neural networks (RNNs) and standard LSTMs process sequential time-series solely in chronological order ($t_1 \to t_T$), pharmaceutical consumption trends frequently exhibit complex temporal dependencies that benefit from dual-directional contextual awareness.

In the HVTM CARE prediction architecture, the **Bidirectional Long Short-Term Memory (BiLSTM)** network processes historical inventory and demand sequences simultaneously in forward and backward temporal directions. This dual-perspective representation allows the model to capture:
- Pre-stockout acceleration trends (consumption surges preceding stock depletion).
- Post-replenishment stabilization behaviors (gradual normalization of buffer stocks).
- Asymmetric macroeconomic responses to foreign exchange rate movements.

---

## 2. Mathematical Formulation

### 2.1 LSTM Cell Mechanics
At each time step $t$, the standard LSTM cell maintains a hidden state vector $h_t$ and an internal cell state vector $c_t$, modulated by three multiplicative gates:

$$\begin{aligned}
\text{Forget Gate:} \quad f_t &= \sigma\left(W_f x_t + U_f h_{t-1} + b_f\right) \\
\text{Input Gate:} \quad i_t &= \sigma\left(W_i x_t + U_i h_{t-1} + b_i\right) \\
\text{Candidate State:} \quad \tilde{c}_t &= \tanh\left(W_c x_t + U_c h_{t-1} + b_c\right) \\
\text{Cell State Update:} \quad c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
\text{Output Gate:} \quad o_t &= \sigma\left(W_o x_t + U_o h_{t-1} + b_o\right) \\
\text{Hidden State:} \quad h_t &= o_t \odot \tanh(c_t)
\end{aligned}$$

Where $\sigma(\cdot)$ represents the sigmoid activation function and $\odot$ denotes the Hadamard element-wise product.

### 2.2 Bidirectional Processing
The BiLSTM layer instantiates two independent hidden state sequences:
1. **Forward Path ($\overrightarrow{h_t}$):** Computes representations moving chronologically from $t = 1$ to $t = T$.
2. **Backward Path ($\overleftarrow{h_t}$):** Computes representations moving in reverse chronological order from $t = T$ to $t = 1$.

The bidirectional hidden representation $H_t$ at any time step is the concatenation of both directional vectors:
$$H_t = \begin{bmatrix} \overrightarrow{h_t} \\ \overleftarrow{h_t} \end{bmatrix}$$

For single-step multi-horizon forecasting, the final concatenated context vector $H_T$ is fed into downstream fully connected dense layers.

---

## 3. Network Topology & Training Specification

The BiLSTM architecture is compiled with deep sequence capacity and non-linear dense projection layers (`notebooks/Deep_Learning_Training.ipynb`):

```
Input Sequence: [Batch Size, 12 Time Steps, 38 Features]
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Bidirectional LSTM Layer 1 (128 Units: 64 Fwd / 64 Bwd) │
│ - Activation: tanh | Recurrent Activation: sigmoid    │
│ - Return Sequences: True                              │
└────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Spatial Dropout Layer (Rate: 0.25)                     │
└────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Bidirectional LSTM Layer 2 (64 Units: 32 Fwd / 32 Bwd)  │
│ - Return Sequences: False (Extracts Final Context)    │
└────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Fully Connected Dense Layer (32 Units, ReLU)           │
└────────────────────────────────────────────────────────┘
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│ Dense Output Layer (1 Unit, Linear Activation)         │
└────────────────────────────────────────────────────────┘
```

### Hyperparameter Configuration

| Parameter | Specification | Purpose |
| :--- | :--- | :--- |
| **Input Shape** | `(12, 38)` | 12-month sequential history across 38 standardized inventory covariates |
| **Total Parameters** | `132,481` | High representative capacity for subtle multi-brand trend variations |
| **Optimizer** | `Adam(learning_rate=0.001)` | Adaptive moment estimation ensuring smooth loss surface convergence |
| **Loss Function** | `Huber Loss` ($\delta = 1.0$) | Robust against extreme drug consumption outliers and supply shocks |
| **Early Stopping** | Patience = `15` epochs | Monitored on validation loss to prevent temporal overfitting |

---

## 4. Role within NeuroStack Meta-Learner
The continuous inference output from the BiLSTM network ($p_{\text{BiLSTM}}$) is serialized and forwarded to the Level-1 **NeuroStack Meta-Learner**:
$$S = \begin{bmatrix} p_{\text{LR}} & p_{\text{RF}} & p_{\text{XGB}} & p_{\text{BiLSTM}} & p_{\text{GRU}} \end{bmatrix}$$

While tree-based regressors (XGBoost, Random Forest) excel at static boundary conditions, the BiLSTM base model provides the ensemble with crucial long-range sequential inertia, ensuring forecasts remain physically realistic during prolonged multi-month supply chain disruptions.
