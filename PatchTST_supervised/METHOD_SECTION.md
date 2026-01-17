# Method Section for Poster

## PatchTST with Weekend/Weekday Embedding for Electricity Load Forecasting

### Motivation
Electricity consumption patterns exhibit distinct characteristics between weekdays and weekends. Traditional time series forecasting models often fail to capture these temporal dependencies, leading to suboptimal predictions. We propose an enhanced PatchTST model that incorporates weekend/weekday embeddings to better model the periodic patterns in electricity consumption.

### Proposed Method

#### 1. PatchTST Architecture
- **Patch-based Processing**: Divides time series into patches of fixed length with stride
- **Channel Independence**: Each variable (electricity meter) is processed independently
- **Transformer Encoder**: Multi-head self-attention mechanism captures temporal dependencies
- **RevIN Normalization**: Reversible instance normalization for better generalization

#### 2. Weekend/Weekday Embedding
- **Temporal Feature Extraction**: 
  - Extract weekday information from timestamps (Monday=0, Sunday=6)
  - Classify as weekday (Mon-Fri: 0) or weekend (Sat-Sun: 1)
  
- **Patch-level Aggregation**:
  - Map sequence-level weekend flags to patch-level representation
  - Use majority voting within each patch to determine weekend/weekday label
  
- **Embedding Integration**:
  - Learnable embedding layer: `E: {0, 1} → R^d` where d is the model dimension
  - Add weekend embedding to positional encoding: `h = PE(pos) + E(weekend_flag)`
  - Enables the model to distinguish between weekday and weekend consumption patterns

#### 3. Model Architecture

```
Input Sequence [B, L, C]
    ↓
RevIN Normalization
    ↓
Patching [B, C, L] → [B, C, patch_num, patch_len]
    ↓
Patch Embedding + Positional Encoding + Weekend Embedding
    ↓
Transformer Encoder (Multi-head Self-Attention)
    ↓
Flatten Head
    ↓
RevIN Denormalization
    ↓
Output [B, pred_len, C]
```

### Key Contributions
1. **Temporal Awareness**: Explicit modeling of weekday/weekend patterns through learnable embeddings
2. **Patch-level Integration**: Weekend information aggregated at patch level for consistency with PatchTST architecture
3. **Minimal Overhead**: Only adds a small embedding layer (2 × d_model parameters) to the base model

### Mathematical Formulation

Given a time series sequence **x** = [x₁, x₂, ..., xₜ] and corresponding weekday information **w** = [w₁, w₂, ..., wₜ] where wᵢ ∈ {0, 1}:

1. **Patching**: 
   - **x** → **P** = [P₁, P₂, ..., Pₙ] where each patch Pᵢ has length `patch_len`
   - **w** → **w_patch** = [w̄₁, w̄₂, ..., w̄ₙ] where w̄ᵢ = majority(wᵢ, ..., wᵢ+patch_len-1)

2. **Embedding**:
   - Patch embedding: **P_emb** = W_P · **P**
   - Positional encoding: **PE** = PE(pos)
   - Weekend embedding: **WE** = E(**w_patch**)
   - Combined: **h** = **P_emb** + **PE** + **WE**

3. **Encoder**:
   - **z** = Transformer(**h**)

4. **Prediction**:
   - **ŷ** = Head(**z**)

### Implementation Details
- **Embedding Dimension**: Same as model dimension (d_model = 128)
- **Embedding Classes**: 2 (weekday=0, weekend=1)
- **Patch Aggregation**: Average-based decision (mean > 0.5 → weekend)
- **Integration**: Additive combination with positional encoding

## 3. Experimental Settings

### 3.1 Dataset Configuration

| Setting | Value |
|---------|-------|
| Dataset | Electricity |
| Number of Variables | 321 |
| Input Length (seq_len) | 336 |
| Prediction Length (pred_len) | 96, 192, 336, 720 |
| Data Split | Train: 70%, Val: 10%, Test: 20% |
| Feature Mode | Multivariate (M) |

### 3.2 Model Hyperparameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| Encoder layers (e_layers) | 3 | Number of Transformer Encoder layers |
| Attention heads (n_heads) | 16 | Number of multi-head attention heads |
| Model dimension (d_model) | 128 | Embedding dimension |
| Feed-forward dimension (d_ff) | 256 | Feed-forward network dimension |
| Dropout | 0.2 | General dropout rate |
| FC dropout | 0.2 | Fully connected layer dropout |
| Head dropout | 0.0 | Head layer dropout |
| Patch length (patch_len) | 16 | Length of each patch |
| Stride | 8 | Stride for patch generation |
| Padding patch | end | Patch padding method |
| RevIN | True | Use Reversible Instance Normalization |
| Weekend embedding | True | Use weekend/weekday embedding |

### 3.3 Training Configuration

| Setting | Value |
|---------|-------|
| Batch size | 16 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Learning rate scheduler | OneCycleLR (pct_start=0.2) |
| Training epochs | 100 |
| Early stopping patience | 10 |
| Random seed | 2021 |

