# 3D Model Design for FRBayes JAX

## Overview

This document outlines two primary approaches for implementing full 3D (frequency × time) models for FRB waterfall data analysis. These models preserve the complete frequency-time structure without collapsing dimensions, enabling more accurate parameter inference and physical interpretation.

## Motivation

Current 2D models apply a simple power-law scaling `(f/f_ref)^α` to collapse the frequency dimension, which loses important information:
- Channel-to-channel intensity variations
- Frequency-dependent pulse broadening (scattering)
- Spectral features (scintillation, absorption)
- RFI and instrumental effects

The 3D models address these limitations by modeling the full waterfall structure.

## Option 2: Basis Function Decomposition

### Concept
Model the spectral (frequency) direction using a set of basis functions, allowing flexible representation of complex spectral shapes while keeping the parameter count manageable.

### Mathematical Framework

The 3D model is constructed as:
```
Model_3D(t, f) = Temporal(t, θ_temporal) × Spectral(f, θ_spectral)
```

Where the spectral component is:
```
Spectral(f) = Σ_k c_k × B_k(f)
```

### Shared Spectrum Approach

**Key Design Decision**: All pulses share the same spectral shape. This assumes:
- All pulses originate from the same emission region
- Similar emission mechanisms across the burst
- Spectral variations affect all components equally

This choice significantly reduces parameters from K×N to just K spectral coefficients.

### Basis Function Choices

1. **Chebyshev Polynomials (Recommended)**
   ```python
   def chebyshev_basis(freq, k, freq_min, freq_max):
       # Map frequency to [-1, 1]
       x = 2 * (freq - freq_min) / (freq_max - freq_min) - 1

       # Recursive computation for numerical stability
       if k == 0:
           return jnp.ones_like(x)
       elif k == 1:
           return x
       else:
           T_prev2 = jnp.ones_like(x)
           T_prev1 = x
           for i in range(2, k+1):
               T_current = 2 * x * T_prev1 - T_prev2
               T_prev2 = T_prev1
               T_prev1 = T_current
           return T_current
   ```
   - **Orthogonal**: Reduces parameter correlation
   - **Stable**: Minimizes Runge phenomenon
   - **Efficient**: Fast convergence for smooth functions
   - **Recommended order**: 3-7 coefficients typically sufficient

2. **Legendre Polynomials**
   ```python
   B_k(f) = P_k((2f - f_min - f_max) / (f_max - f_min))
   ```
   - Alternative orthogonal basis
   - Similar properties to Chebyshev

3. **Simple Polynomials**
   ```python
   B_k(f) = ((f - f_center) / f_scale)^k
   ```
   - Simpler but less stable
   - Not recommended for k > 4

### Parameter Structure (Shared Spectrum)

For N pulses and K basis functions with shared spectrum:
```
θ = [
    # Temporal parameters (frequency-independent)
    A_1, ..., A_N,        # Pulse amplitudes (reference)
    τ_1, ..., τ_N,        # Decay times
    u_1, ..., u_N,        # Arrival times
    w_1, ..., w_N,        # Widths (EMG only)

    # Spectral parameters (shared across all pulses)
    c_0, ..., c_{K-1},    # Basis function coefficients

    # Noise
    σ                     # Noise level
]
```

**Total parameters**: 4N + K + 1 (for EMG)
- Compare to per-pulse spectra: 4N + K×N + 1
- Reduction: K×(N-1) fewer parameters

### Implementation Example (Shared Spectrum)

```python
@jit
def emg_model_3d_basis(t, freq, theta, max_peaks, n_basis, fit_pulses=False):
    """
    3D EMG model using basis function decomposition with shared spectrum.
    All pulses share the same spectral shape.
    """
    # Extract temporal parameters
    n_temporal = 4 * max_peaks  # A, tau, u, w for each peak
    temporal_params = theta[:n_temporal]

    # Extract spectral coefficients (shared for all pulses)
    spectral_coeffs = theta[n_temporal:n_temporal + n_basis]

    # Extract sigma
    sigma_idx = n_temporal + n_basis
    sigma = theta[sigma_idx]

    # Optional: Npulse parameter
    if fit_pulses:
        npulse = jnp.round(theta[-1])
    else:
        npulse = max_peaks

    # Compute temporal model (1D) - sum of all pulses
    temporal_model = jnp.zeros_like(t)
    for i in range(max_peaks):
        if i < npulse:
            A = temporal_params[i]
            tau = temporal_params[max_peaks + i]
            u = temporal_params[2*max_peaks + i]
            w = temporal_params[3*max_peaks + i]
            temporal_model += emg_pulse(t, A, tau, u, w)

    # Compute spectral model using Chebyshev basis
    freq_min, freq_max = jnp.min(freq), jnp.max(freq)
    spectral_model = jnp.zeros_like(freq)
    for k in range(n_basis):
        spectral_model += spectral_coeffs[k] * chebyshev_basis(freq, k, freq_min, freq_max)

    # Ensure positive spectral scaling
    spectral_model = jnp.exp(spectral_model)  # Or use softplus for smoother behavior

    # Combine: outer product gives (freq, time) array
    model_3d = spectral_model[:, None] * temporal_model[None, :]

    return model_3d
```

### Per-Pulse vs Shared Spectrum Trade-offs

| Aspect | Shared Spectrum | Per-Pulse Spectra |
|--------|----------------|-------------------|
| **Parameters** | 4N + K + 1 | 4N + K×N + 1 |
| **Flexibility** | Moderate | High |
| **Overfitting Risk** | Lower | Higher |
| **Physical Assumption** | Common emission | Different mechanisms |
| **Best For** | Periodic FRBs, single source | Complex multi-component |

### Model Selection Strategy

```python
# Start with shared spectrum
model_shared = emg_model_3d_basis(shared_spectrum=True)
evidence_shared = run_nested_sampling_3d(model_shared)

# Test if per-pulse spectra needed
if residuals show frequency-dependent structure per pulse:
    model_perpulse = emg_model_3d_basis(shared_spectrum=False)
    evidence_perpulse = run_nested_sampling_3d(model_perpulse)

    # Bayesian model comparison
    if log(evidence_perpulse) - log(evidence_shared) > 5:
        use_per_pulse = True
```

### Prior Recommendations for Basis Coefficients

1. **Uniform priors**: `c_k ~ Uniform(-2, 2)`
   - Simple, non-informative
   - Works well for first few coefficients

2. **Hierarchical regularization**:
   ```python
   c_0 ~ Uniform(-2, 2)  # Zeroth order less constrained
   c_k ~ Normal(0, σ_k) where σ_k = σ_0 * 0.5^k  # Decay for higher orders
   ```
   - Natural regularization
   - Prevents high-order oscillations

### Advantages
- **Flexible**: Can represent arbitrary spectral shapes
- **Efficient**: Fewer parameters than independent channels
- **Smooth**: Natural regularization through basis functions
- **Interpretable**: Coefficients show spectral structure
- **Stable**: Chebyshev basis avoids numerical issues

### Disadvantages
- **Basis selection**: Must choose appropriate basis for data
- **Non-physical**: Parameters may lack direct physical meaning
- **Shared spectrum limitation**: Cannot model different spectral evolution per pulse

---

## Option 3: Physical Template Model

### Concept
Model frequency-dependent effects based on known astrophysical processes, with each parameter having direct physical meaning.

### Mathematical Framework

Each pulse has frequency-dependent parameters following physical laws:

```
Pulse_i(t, f) = A_i(f) × Shape(t; τ_i(f), u_i(f), w_i(f))
```

Where:
- `A_i(f) = A_0i × (f/f_ref)^α_i` - Power-law spectral index
- `τ_i(f) = τ_0i × (f/f_ref)^(-4+ε)` - Scattering broadening
- `u_i(f) = u_0i` - Arrival time (constant for de-dispersed data)
- `w_i(f) = w_0i × (f/f_ref)^γ` - Intrinsic width scaling

### Physical Effects Modeled

1. **Spectral Index (α)**
   - Origin: Emission mechanism (synchrotron, curvature radiation)
   - Range: Typically -5 to +2
   - Can be per-pulse or shared

2. **Scattering (τ_scatter)**
   - Origin: Propagation through turbulent ISM
   - Scaling: ∝ f^-4 for Kolmogorov turbulence
   - Causes one-sided (exponential) broadening

3. **Intrinsic Width Variation (γ)**
   - Origin: Emission region size, beaming
   - Often negligible but can be included

### Parameter Structure

For N pulses with physical scaling:
```
θ = [
    # Reference amplitudes
    A_01, ..., A_0N,      # Amplitudes at reference frequency

    # Spectral indices
    α_1, ..., α_N,        # Per-pulse spectral indices
    # OR
    α_shared,             # Single shared spectral index

    # Scattering parameters
    τ_01, ..., τ_0N,      # Reference scattering times
    ε,                    # Scattering index deviation

    # Temporal parameters (frequency-independent)
    u_1, ..., u_N,        # Arrival times
    w_01, ..., w_0N,      # Reference widths

    # Optional width scaling
    γ,                    # Width frequency scaling

    # Noise
    σ                     # Noise level
]
```

Total parameters: ~5N + 3 (with shared indices)

### Implementation Example

```python
def emg_model_3d_physical(t, freq, theta, max_peaks, ref_freq=1400.0):
    """
    3D EMG model with physically motivated frequency scaling.
    """
    # Extract parameters
    A_0 = theta[:max_peaks]                    # Reference amplitudes
    alpha = theta[max_peaks]                   # Spectral index (shared)
    tau_0 = theta[max_peaks+1:2*max_peaks+1]   # Reference scattering
    epsilon = theta[2*max_peaks+1]             # Scattering deviation
    u = theta[2*max_peaks+2:3*max_peaks+2]     # Arrival times
    w_0 = theta[3*max_peaks+2:4*max_peaks+2]   # Reference widths

    # Initialize 3D model
    model_3d = jnp.zeros((len(freq), len(t)))

    # For each frequency channel
    def compute_channel(f):
        # Frequency scaling
        f_ratio = f / ref_freq

        # Scale parameters
        A_scaled = A_0 * f_ratio**alpha
        tau_scaled = tau_0 * f_ratio**(-4 + epsilon)
        w_scaled = w_0  # Can add scaling if needed

        # Compute model for this frequency
        channel_model = jnp.zeros_like(t)
        for i in range(max_peaks):
            pulse = emg_pulse(t, A_scaled[i], tau_scaled[i], u[i], w_scaled[i])
            channel_model += pulse

        return channel_model

    # Vectorize over frequency
    model_3d = vmap(compute_channel)(freq)

    return model_3d
```

### Advantages
- **Physical interpretation**: Every parameter has meaning
- **Predictive power**: Can extrapolate to other frequencies
- **Constrained**: Physical laws reduce parameter space
- **Scientific insight**: Reveals emission/propagation physics

### Disadvantages
- **Model assumptions**: May not capture all effects
- **Less flexible**: Cannot fit arbitrary spectral shapes
- **Complex implementation**: Multiple scaling laws

---

## Comparison

| Aspect | Basis Function | Physical Template |
|--------|---------------|-------------------|
| **Parameter Count** | 4N + K + 1 | ~5N + 3 |
| **Flexibility** | Very high | Moderate |
| **Physical Meaning** | Low | High |
| **Implementation Complexity** | Simple | Moderate |
| **Computational Cost** | Low | Moderate |
| **Overfitting Risk** | Higher | Lower |
| **Scientific Value** | Descriptive | Predictive |
| **Best For** | Complex spectra, exploration | Physical parameter extraction |

## Implementation Recommendations

### Start Simple (Minimal Physical Model)
```python
# Phase 1: Just spectral index
θ = [A_01...A_0N, α, τ_1...τ_N, u_1...u_N, w_1...w_N, σ]

# Phase 2: Add scattering
θ = [A_01...A_0N, α, τ_01...τ_0N, ε, u_1...u_N, w_1...w_N, σ]

# Phase 3: Full model if needed
θ = [full physical parameter set]
```

### Validation Strategy
1. **Synthetic data**: Test with known parameters
2. **Residual analysis**: Check for systematic frequency trends
3. **Model comparison**: Use Bayesian evidence to select complexity
4. **Physical consistency**: Verify parameters are in expected ranges

### Hybrid Approach
Consider combining both approaches:
- Use physical model for main signal
- Add basis functions for residual features
- Best of both worlds: interpretability + flexibility

## Code Integration Path

1. **Add to `models.py`**:
   - `emg_model_3d_minimal()` - Start here
   - `emg_model_3d_physical()` - Full physical model
   - `emg_model_3d_basis()` - Basis function approach

2. **Update `priors.py`**:
   - Add bounds for new parameters (α, ε, γ, β)
   - Create `FRBPriors3D` class

3. **Extend `sampling.py`**:
   - `run_nested_sampling_3d()` function
   - Efficient likelihood for 3D data

4. **Enhance `analysis.py`**:
   - Frequency-resolved plots
   - Parameter evolution diagnostics

## Next Steps

1. Implement minimal physical model first
2. Validate on synthetic data
3. Test on real FRB 20191221A data
4. Add complexity only if Bayesian evidence improves
5. Document physical interpretations of fitted parameters