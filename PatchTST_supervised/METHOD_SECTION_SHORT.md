# Method Section (Short Version for Poster)

## PatchTST with Weekend/Weekday Embedding

### Problem
Electricity consumption patterns differ significantly between weekdays and weekends, but existing models fail to explicitly model this temporal dependency.

### Solution
We enhance PatchTST by incorporating learnable weekend/weekday embeddings that are integrated with positional encodings.

### Method

**1. Temporal Feature Extraction**
- Extract weekday information from timestamps
- Classify as weekday (Mon-Fri) or weekend (Sat-Sun)

**2. Patch-level Weekend Embedding**
- Aggregate weekend flags at patch level using majority voting
- Learnable embedding: `E: {weekday, weekend} → R^d`
- Add to positional encoding: `h = PE(pos) + E(weekend_flag)`

**3. Architecture**
```
Input → RevIN → Patching → [Patch Embedding + PE + Weekend Embedding] 
→ Transformer Encoder → Head → RevIN⁻¹ → Output
```

### Key Innovation
- **Explicit temporal modeling**: Learnable embeddings capture weekday/weekend patterns
- **Minimal overhead**: Only adds 2 × d_model parameters
- **Patch-level integration**: Consistent with PatchTST's patch-based architecture










