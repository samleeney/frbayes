"""
BlackJAX nested sampling implementation for FRBayes
"""

import jax
import jax.numpy as jnp
from jax import random
import blackjax
from blackjax.nss import finalise
import numpy as np
import time
from typing import Dict, Callable, Any, Tuple


class BlackJAXNestedSampler:
    """
    Nested sampling using BlackJAX
    """
    
    def __init__(self, 
                 model,
                 data: Dict[str, jnp.ndarray],
                 prior_config: Dict = None,
                 config: Dict = None):
        """
        Initialize the nested sampler
        
        Args:
            model: Model object with log_likelihood and sample_prior methods
            data: Dictionary containing observed data
            prior_config: Prior configuration dictionary
            config: Sampler configuration
        """
        self.model = model
        self.data = data
        self.prior_config = prior_config or {}
        self.config = config or {}
        
        # Set up log likelihood function
        self.log_likelihood_fn = lambda theta: self.log_likelihood(theta)
        self.log_prior_fn = lambda theta: self.log_prior(theta)
    
    def log_likelihood(self, theta: jnp.ndarray) -> float:
        """
        Compute log likelihood for flattened parameter vector
        
        Args:
            theta: Flattened parameter array
            
        Returns:
            Log likelihood value
        """
        # Unflatten parameters
        params = self.unflatten_params(theta)
        
        # Get model prediction
        model_pred = self.model.model_function(self.data['t'], params)
        
        # Compute residuals
        residuals = self.data['pp'] - model_pred
        
        # Get noise parameter
        sigma = params['sigma'][0]
        
        # Compute log likelihood (Gaussian)
        log_L = -0.5 * jnp.sum((residuals / sigma) ** 2)
        log_L -= 0.5 * len(residuals) * jnp.log(2 * jnp.pi * sigma**2)
        
        return log_L
    
    def log_prior(self, theta: jnp.ndarray) -> float:
        """
        Compute log prior for flattened parameter vector
        
        Args:
            theta: Flattened parameter array
            
        Returns:
            Log prior value
        """
        # Unflatten parameters
        params = self.unflatten_params(theta)
        
        log_prob = 0.0
        
        # Amplitude (uniform)
        A_range = self.prior_config.get('amplitude', {'min': 0.001, 'max': 0.1})
        for a in params['A']:
            if a < A_range['min'] or a > A_range['max']:
                return -jnp.inf
            log_prob -= jnp.log(A_range['max'] - A_range['min'])
        
        # Tau (uniform)
        tau_range = self.prior_config.get('tau', {'min': 0.1, 'max': 1.0})
        for tau in params['tau']:
            if tau < tau_range['min'] or tau > tau_range['max']:
                return -jnp.inf
            log_prob -= jnp.log(tau_range['max'] - tau_range['min'])
        
        # u (uniform)
        u_range = self.prior_config.get('u', {'min': 0.0, 'max': 5.0})
        for u in params['u']:
            if u < u_range['min'] or u > u_range['max']:
                return -jnp.inf
            log_prob -= jnp.log(u_range['max'] - u_range['min'])
        
        # width (uniform)
        w_range = self.prior_config.get('width', {'min': 0.05, 'max': 0.5})
        for w in params['w']:
            if w < w_range['min'] or w > w_range['max']:
                return -jnp.inf
            log_prob -= jnp.log(w_range['max'] - w_range['min'])
        
        # sigma (log-uniform)
        sigma_range = self.prior_config.get('sigma', {'min': 1e-4, 'max': 0.01})
        sigma = params['sigma'][0]
        if sigma < sigma_range['min'] or sigma > sigma_range['max']:
            return -jnp.inf
        log_prob -= jnp.log(jnp.log(sigma_range['max'] / sigma_range['min']))
        log_prob -= jnp.log(sigma)
        
        # Npulse (uniform) if fit_pulses is True
        if 'Npulse' in params:
            Npulse_range = self.prior_config.get('Npulse', {'min': 1, 'max': self.model.max_peaks})
            Npulse = params['Npulse'][0]
            Npulse_in_bounds = (Npulse >= Npulse_range['min']) & (Npulse <= Npulse_range['max'])
            log_prob = jnp.where(
                Npulse_in_bounds,
                log_prob - jnp.log(Npulse_range['max'] - Npulse_range['min']),
                -jnp.inf
            )
        
        return log_prob
    
    def sample_from_prior(self, key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """
        Sample from the prior distribution
        
        Args:
            key: JAX random key
            n_samples: Number of samples to generate
            
        Returns:
            Array of samples from the prior
        """
        samples = []
        
        for i in range(n_samples):
            key, subkey = random.split(key)
            params = {}
            
            # Sample amplitude (uniform)
            A_range = self.prior_config.get('amplitude', {'min': 0.001, 'max': 0.1})
            key, subkey = random.split(key)
            params['A'] = random.uniform(
                subkey, 
                shape=(self.model.max_peaks,),
                minval=A_range['min'],
                maxval=A_range['max']
            )
            
            # Sample tau (uniform)
            tau_range = self.prior_config.get('tau', {'min': 0.1, 'max': 1.0})
            key, subkey = random.split(key)
            params['tau'] = random.uniform(
                subkey,
                shape=(self.model.max_peaks,),
                minval=tau_range['min'],
                maxval=tau_range['max']
            )
            
            # Sample u (uniform)
            u_range = self.prior_config.get('u', {'min': 0.0, 'max': 5.0})
            key, subkey = random.split(key)
            params['u'] = random.uniform(
                subkey,
                shape=(self.model.max_peaks,),
                minval=u_range['min'],
                maxval=u_range['max']
            )
            
            # Sample width (uniform)
            w_range = self.prior_config.get('width', {'min': 0.05, 'max': 0.5})
            key, subkey = random.split(key)
            params['w'] = random.uniform(
                subkey,
                shape=(self.model.max_peaks,),
                minval=w_range['min'],
                maxval=w_range['max']
            )
            
            # Sample sigma (log-uniform)
            sigma_range = self.prior_config.get('sigma', {'min': 1e-4, 'max': 0.01})
            key, subkey = random.split(key)
            log_sigma = random.uniform(
                subkey,
                shape=(1,),
                minval=jnp.log(sigma_range['min']),
                maxval=jnp.log(sigma_range['max'])
            )
            params['sigma'] = jnp.exp(log_sigma)
            
            # Sample Npulse if fit_pulses is True
            if self.model.fit_pulses:
                Npulse_range = self.prior_config.get('Npulse', {'min': 1, 'max': self.model.max_peaks})
                key, subkey = random.split(key)
                params['Npulse'] = random.uniform(
                    subkey,
                    shape=(1,),
                    minval=Npulse_range['min'],
                    maxval=Npulse_range['max']
                )
            
            # Flatten and add to samples
            samples.append(self.flatten_params(params))
        
        return jnp.array(samples)
    
    def flatten_params(self, params: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Flatten parameter dictionary to array"""
        theta = []
        
        # Follow consistent ordering
        for key in ['A', 'tau', 'u']:
            if key in params:
                theta.append(params[key].flatten())
        
        if 'w' in params:
            theta.append(params['w'].flatten())
        
        if 'sigma' in params:
            theta.append(params['sigma'].flatten())
            
        if 'Npulse' in params:
            theta.append(params['Npulse'].flatten())
        
        return jnp.concatenate(theta)
    
    def unflatten_params(self, theta: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Unflatten parameter array to dictionary"""
        params = {}
        idx = 0
        
        # Extract parameters in order
        n = self.model.max_peaks
        
        params['A'] = theta[idx:idx+n]
        idx += n
        
        params['tau'] = theta[idx:idx+n]
        idx += n
        
        params['u'] = theta[idx:idx+n]
        idx += n
        
        params['w'] = theta[idx:idx+n]
        idx += n
        
        params['sigma'] = theta[idx:idx+1]
        idx += 1
        
        if self.model.fit_pulses:
            params['Npulse'] = theta[idx:idx+1]
        
        return params
    
    def run(self, 
            key: jax.random.PRNGKey,
            num_live_points: int = 1000,
            max_samples: int = 100000,
            precision_criterion: float = 0.01) -> Dict:
        """
        Run BlackJAX nested sampling
        
        Args:
            key: JAX random key
            num_live_points: Number of live points
            max_samples: Maximum number of iterations (default very high to avoid early termination)
            precision_criterion: Stopping criterion (log(Z) error threshold)
            
        Returns:
            Dictionary with samples and diagnostics
        """
        print(f"Running BlackJAX nested sampling with {num_live_points} live points...")
        start_time = time.time()
        
        # Initialize the nested sampling algorithm
        algo = blackjax.nss(
            logprior_fn=self.log_prior_fn,
            loglikelihood_fn=self.log_likelihood_fn,
            num_delete=50,  # Number of points to delete per iteration
            num_inner_steps=20,  # Number of slice sampling steps
        )
        
        # Sample initial live points from prior
        key, init_key = random.split(key)
        initial_live_points = self.sample_from_prior(init_key, num_live_points)
        
        # Initialize state
        state = algo.init(initial_live_points)
        
        # Define JIT-compiled step function
        @jax.jit
        def one_step(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, info = algo.step(subk, state)
            return (state, k), info
        
        # Run the sampler
        print("Starting nested sampling iterations...")
        
        dead_points = []
        iteration = 0
        
        # Main sampling loop
        # Following https://handley-lab.co.uk/nested-sampling-book/basic/quickstart.html
        # The run terminates when log(Z_live) - log(Z) < -precision_criterion
        # Default in documentation is -3, but we allow it to be configurable
        while iteration < max_samples:
            # Check convergence: terminate when remaining live points contribute negligibly
            logZ_diff = state.logZ_live - state.logZ
            if logZ_diff < -abs(precision_criterion):
                print(f"Converged after {iteration} iterations")
                print(f"log(Z) = {state.logZ:.4f}")
                print(f"log(Z_live) - log(Z) = {logZ_diff:.4f} < -{abs(precision_criterion)}")
                break
            
            # Take steps (delete 50 points at a time)
            (state, key), dead_info = one_step((state, key), None)
            dead_points.append(dead_info)
            
            iteration += 50  # We delete 50 points per iteration
            
            # Print progress
            if iteration % 500 == 0:
                print(f"Iteration {iteration}: log(Z) = {state.logZ:.2f}, "
                      f"log(Z_live) = {state.logZ_live:.2f}")
        
        elapsed = time.time() - start_time
        print(f"Sampling completed in {elapsed:.1f} seconds")
        
        # Finalize the sampling
        final_info = finalise(state, dead_points)
        
        # The final log evidence is in the state
        final_log_Z = state.logZ
        print(f"Final log(Z) = {final_log_Z:.4f}")
        
        # Process and return results
        results = self.process_results(
            final_info=final_info,
            final_log_Z=final_log_Z,
            num_iterations=iteration,
            final_state=state  # Pass the full state for anesthetic
        )
        
        results['elapsed_time'] = elapsed
        results['num_live_points'] = num_live_points
        
        return results
    
    def process_results(self, final_info, final_log_Z, num_iterations, final_state=None) -> Dict:
        """
        Process nested sampling results
        
        Args:
            final_info: Final NSInfo from BlackJAX
            final_log_Z: Final log evidence
            num_iterations: Number of iterations
            final_state: Final nested sampling state (for anesthetic)
            
        Returns:
            Dictionary with processed results
        """
        # Extract samples and weights from NSInfo
        samples = np.array(final_info.particles)
        log_weights = np.array(final_info.log_weights)
        
        # Normalize weights
        log_weights_normalized = log_weights - jax.scipy.special.logsumexp(log_weights)
        weights = np.exp(log_weights_normalized)
        
        # Sample from posterior using weights
        num_samples = min(1000, len(samples))
        
        # Random weighted resampling
        key = random.PRNGKey(42)
        
        # Weighted resampling
        indices = random.choice(
            key,
            len(samples),
            shape=(num_samples,),
            p=weights
        )
        
        # Convert to parameter dictionaries
        posterior_samples = []
        for idx in indices:
            theta = samples[idx]
            params = self.unflatten_params(theta)
            posterior_samples.append(params)
        
        results = {
            'samples': posterior_samples,
            'weights': weights[:num_samples],  # Match number of samples
            'log_weights': log_weights_normalized[:num_samples],
            'log_evidence': final_log_Z,
            'num_iterations': num_iterations,
            'all_samples': samples,  # Keep all samples for diagnostics
            'all_log_weights': log_weights_normalized,
        }
        
        # Add raw state info for anesthetic if available
        # Store the final_info object directly (result of finalise())
        if final_info is not None:
            # Replace NaN values in logL_birth with -inf for anesthetic compatibility
            # The final live points have NaN logL_birth since they never died
            import numpy as np
            logL_birth_fixed = np.array(final_info.loglikelihood_birth)
            logL_birth_fixed = np.where(np.isnan(logL_birth_fixed), -np.inf, logL_birth_fixed)
            
            # Create a modified final_info with fixed logL_birth
            from collections import namedtuple
            NSInfo = namedtuple('NSInfo', ['particles', 'loglikelihood', 'loglikelihood_birth'])
            final_info_fixed = NSInfo(
                particles=final_info.particles,
                loglikelihood=final_info.loglikelihood,
                loglikelihood_birth=logL_birth_fixed
            )
            results['final_info'] = final_info_fixed  # Full info including live points
            
            # For anesthetic: provide clean dead-points-only data (Option 1)
            # The final live points have NaN logL_birth since they never died
            particles = np.array(final_info.particles)
            logL = np.array(final_info.loglikelihood)
            logL_birth = np.array(final_info.loglikelihood_birth)
            
            # Extract only dead points (those with valid logL_birth)
            valid_mask = ~np.isnan(logL_birth)
            results['particles'] = particles  # Keep all for reference
            results['logL'] = logL
            results['logL_birth'] = logL_birth
            
            # Clean versions for anesthetic (dead points only)
            results['particles_dead'] = particles[valid_mask]
            results['logL_dead'] = logL[valid_mask]
            results['logL_birth_dead'] = logL_birth[valid_mask]
            
            # Store counts
            results['n_dead'] = np.sum(valid_mask)
            results['n_live_final'] = np.sum(~valid_mask)
        
        # Also store the final state if provided
        if final_state is not None:
            results['final_state'] = final_state
        
        results['log_Z'] = final_log_Z
        results['num_iterations'] = num_iterations
        
        return results