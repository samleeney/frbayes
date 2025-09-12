# Derivation of 2D Likelihood for FRB Fitting with Spectral Index

## 1. Physical Motivation

Fast Radio Bursts (FRBs) exhibit frequency-dependent intensity variations that can be characterized by a power-law spectral index. For dedispersed data, the temporal structure remains constant across frequency channels, but the amplitude scales with frequency.

## 2. Model Formulation

### 2.1 One-Dimensional Temporal Model

For a single frequency channel, the temporal model for multiple pulses is:

$$M_{1D}(t; \boldsymbol{\theta}) = \sum_{i=1}^{N_{peaks}} P_i(t; A_i, \tau_i, u_i, w_i)$$

where:
- $P_i$ is the pulse shape function (exponential or EMG)
- $A_i$ is the amplitude of pulse $i$
- $\tau_i$ is the decay time constant
- $u_i$ is the arrival time
- $w_i$ is the width parameter (for EMG only)

### 2.2 Extension to Two Dimensions

For dedispersed data, we model the frequency dependence using a spectral index $\alpha$:

$$M_{2D}(\nu, t; \boldsymbol{\theta}, \alpha) = M_{1D}(t; \boldsymbol{\theta}) \times \left(\frac{\nu}{\nu_{ref}}\right)^{\alpha}$$

where:
- $\nu$ is the observation frequency
- $\nu_{ref}$ is a reference frequency (typically 1400 MHz)
- $\alpha$ is the spectral index

This formulation assumes:
1. The temporal structure is identical at all frequencies (valid for dedispersed data)
2. Only the amplitude scales with frequency following a power law
3. The spectral index is the same for all pulses

## 3. Likelihood Derivation

### 3.1 Data Model

The observed data $D$ is a 2D array with dimensions $(N_\nu, N_t)$ where:
- $N_\nu$ is the number of frequency channels
- $N_t$ is the number of time bins

For each frequency channel $j$ and time bin $k$:

$$D_{jk} = M_{2D}(\nu_j, t_k; \boldsymbol{\theta}, \alpha) + n_{jk}$$

where $n_{jk}$ is the noise, assumed to be Gaussian with standard deviation $\sigma_j$ (which may be frequency-dependent).

### 3.2 Gaussian Likelihood

Assuming independent Gaussian noise, the likelihood is:

$$\mathcal{L}(\boldsymbol{\theta}, \alpha | D) = \prod_{j=1}^{N_\nu} \prod_{k=1}^{N_t} \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left(-\frac{(D_{jk} - M_{2D}(\nu_j, t_k))^2}{2\sigma_j^2}\right)$$

Taking the logarithm:

$$\ln \mathcal{L} = -\frac{1}{2}\sum_{j=1}^{N_\nu} \sum_{k=1}^{N_t} \left[\frac{(D_{jk} - M_{2D}(\nu_j, t_k))^2}{\sigma_j^2} + \ln(2\pi\sigma_j^2)\right]$$

### 3.3 Simplified Form with Single Noise Parameter

If we assume a single noise level $\sigma$ across all channels:

$$\ln \mathcal{L} = -\frac{1}{2\sigma^2}\sum_{j=1}^{N_\nu} \sum_{k=1}^{N_t} (D_{jk} - M_{2D}(\nu_j, t_k))^2 - N_\nu N_t \ln(\sigma\sqrt{2\pi})$$

### 3.4 Computational Implementation

Expanding the model term:

$$M_{2D}(\nu_j, t_k) = M_{1D}(t_k; \boldsymbol{\theta}) \times \left(\frac{\nu_j}{\nu_{ref}}\right)^{\alpha}$$

The key insight is that $M_{1D}(t_k; \boldsymbol{\theta})$ needs to be computed only once for all time points, then scaled by the frequency-dependent factor for each channel. This is efficiently implemented using JAX's `vmap`:

```python
def scale_by_freq(f):
    return base_model * (f / ref_freq) ** alpha

model_2d = vmap(scale_by_freq)(freq_array)
```

## 4. Parameter Efficiency

### Traditional Approach
Fitting each frequency channel independently would require:
- Parameters per channel: $4N_{peaks}$ (for EMG) or $3N_{peaks}$ (for exponential)
- Total parameters: $N_\nu \times (3\text{ or }4) \times N_{peaks}$

Example: 32 frequency channels, 2 peaks, exponential model = 192 parameters

### Spectral Index Approach
Using the spectral index formulation:
- Temporal parameters: $(3\text{ or }4) \times N_{peaks}$
- Spectral index: 1
- Noise: 1 (or $N_\nu$ for per-channel noise)
- Total: $(3\text{ or }4) \times N_{peaks} + 2$

Example: Same scenario = 8 parameters

**Reduction factor: 24×**

## 5. Physical Interpretation

The spectral index $\alpha$ captures several physical effects:
- **Intrinsic spectrum**: The emission mechanism's frequency dependence
- **Propagation effects**: Frequency-dependent absorption/scattering
- **Typical values**: $\alpha \approx -1.4$ for FRBs (steeper than typical pulsars)

## 6. Advantages of This Formulation

1. **Parameter efficiency**: Dramatically reduces the number of free parameters
2. **Physical motivation**: Spectral index is a meaningful astrophysical quantity
3. **Improved constraints**: All frequency channels contribute to constraining temporal parameters
4. **Computational efficiency**: Model evaluation vectorized over frequency
5. **Noise robustness**: Bad channels naturally down-weighted in likelihood

## 7. Implementation in FRBayes JAX

The implementation leverages JAX's capabilities:
- **JIT compilation**: The entire likelihood is compiled for fast execution
- **Automatic differentiation**: Gradients computed automatically for MCMC
- **GPU acceleration**: Vectorized operations run efficiently on GPUs
- **vmap**: Parallel evaluation across frequency channels

### Code Structure

```python
# In models.py
def exponential_model_2d(t, freq, theta, max_peaks, fit_pulses, ref_freq=1400.0):
    alpha = theta[alpha_idx]
    base_model = exponential_model(t, theta_base, max_peaks, fit_pulses)
    
    def scale_by_freq(f):
        return base_model * (f / ref_freq) ** alpha
    
    model_2d = vmap(scale_by_freq)(freq)
    return model_2d

# In sampling.py (likelihood)
def loglikelihood_fn(theta):
    model_2d = model_func(t_jax, freq_jax, theta, max_peaks, fit_pulses, ref_freq)
    residuals = data_2d_jax - model_2d
    
    if noise_per_channel is not None:
        weighted_residuals = residuals / noise_jax[:, None]
        log_likelihood = -0.5 * jnp.sum(weighted_residuals**2)
    else:
        log_likelihood = -0.5 * jnp.sum((residuals / sigma) ** 2)
        log_likelihood -= n_total * jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
    
    return log_likelihood
```

## 8. Summary

The 2D likelihood with spectral index provides an elegant solution for fitting multi-frequency FRB data:
- Reduces parameters from $O(N_\nu \times N_{peaks})$ to $O(N_{peaks})$
- Maintains physical interpretability through the spectral index
- Leverages all frequency information for improved parameter constraints
- Computationally efficient through vectorization

This formulation is particularly well-suited for dedispersed FRB data where the primary frequency dependence is in the intensity scaling rather than the temporal structure.