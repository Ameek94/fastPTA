# Global imports
import jax
import jax.numpy as jnp

# Local imports
import fastPTA.utils as ut
from fastPTA.inference_tools import signal_covariance as sc
from fastPTA.angular_decomposition import spherical_harmonics as sph


@jax.jit
def log_likelihood_full(parameters, data, gamma_IJ_lm, C_ff):
    """
    Compute the full log-likelihood for the data using a Kronecker product
    structure for the covariance matrix.

    Parameters:
    -----------
    parameters : Array
        Array of power spectrum parameters for the anisotropic signal.
    data : Array
        4D array containing the observed data with shape
        (n_frequencies, n_frequencies, n_pulsars, n_pulsars).
    gamma_IJ_lm : Array
        3D array of spherical harmonics correlations with shape
        (n_coeffs, n_pulsars, n_pulsars), where n_coeffs is the
        number of spherical harmonic coefficients.
    C_ff : Array
        2D array representing the frequency-frequency covariance
        with shape (n_frequencies, n_frequencies).

    Returns:
    --------
    float
        The negative log-likelihood value.
    """

    C_IJ = jnp.einsum("p,pij->ij", parameters, gamma_IJ_lm)

    C_inv = sc.get_inverse_covariance_full(parameters, gamma_IJ_lm, C_ff)

    logdet = ut.logdet_kronecker_product(C_ff, C_IJ)

    data_term = jnp.einsum("mnIJ,nmJI->", C_inv, data)

    return -(logdet + data_term)


# @jax.jit
def log_posterior_full(parameters, Nside, l_max, data, gamma_IJ_lm, C_ff):
    """
    Compute the full log-posterior probability for anisotropic signal.

    This function applies a physical prior that the power spectrum must be
    positive in all pixels of the reconstructed map.

    Parameters:
    -----------
    parameters : Array
        Array of power spectrum parameters for the anisotropic signal.
    Nside : int
        HEALPix Nside parameter controlling the map resolution.
    l_max : int
        Maximum multipole order for the spherical harmonics decomposition.
    data : Array
        4D array containing the observed data with shape
        (n_frequencies, n_frequencies, n_pulsars, n_pulsars).
    gamma_IJ_lm : Array
        3D array of spherical harmonics correlations with shape
        (n_coeffs, n_pulsars, n_pulsars), where n_coeffs is the
        number of spherical harmonic coefficients.
    C_ff : Array
        2D array representing the frequency-frequency covariance
        with shape (n_frequencies, n_frequencies).

    Returns:
    --------
    float
        The log-posterior value, or -infinity if the prior is violated.
    """
    Pk = sph.get_map_from_real_clms(parameters, Nside, l_max=l_max)
    lp = jnp.min(Pk)
    lp = jnp.where(lp < 0, -jnp.inf, lp)

    log_lik = log_likelihood_full(parameters, data, gamma_IJ_lm, C_ff)

    return lp + log_lik


@jax.jit
def log_likelihood(
    data,
    signal_value,
    response_IJ,
    strain_omega,
    response_IJ_V=None,
    signal_value_V=None,
):
    """
    Compute the logarithm of the likelihood assujming a Whittle likelihood.

    When `response_IJ_V` and `signal_value_V` are provided, circular
    polarisation (Stokes V) is included: the data covariance becomes the
    complex Hermitian matrix ``C = R + i A``, with ``R`` the intensity-plus-noise
    covariance and ``A = response_IJ_V * P_V(f)`` the real, antisymmetric V-mode
    signal. The Whittle form ``-sum(logdet C + Tr[C^-1 D])`` is unchanged; it is
    real because ``C`` (and the data ``D``) are Hermitian. With the V arguments
    left as ``None`` this reduces exactly to the intensity-only likelihood.

    Parameters:
    -----------
    data : Array
        Array containing the observed data.
    signal_value : Array
        Array containing the signal evaluated in all frequency bins.
    response_IJ : Array
        Array containing response function.
    strain_omega : Array
        Array containing strain noise.
    response_IJ_V : Array, optional
        Contracted V-mode response, shape (F, N, N), real and antisymmetric in
        the pulsar indices. Default is None.
    signal_value_V : Array, optional
        V-mode signal evaluated in all frequency bins, shape (F,). Required when
        `response_IJ_V` is provided. Default is None.

    Returns:
    --------
    float
        Logarithm of the likelihood.

    """

    # Covariance of the data as signal * response + noise
    covariance = (
        jnp.einsum("ijk,i->ijk", response_IJ, signal_value) + strain_omega
    )

    # Add the circular-polarisation (V-mode) antisymmetric part: C = R + i A
    if response_IJ_V is not None and signal_value_V is not None:
        A = jnp.einsum("ijk,i->ijk", response_IJ_V, signal_value_V)
        covariance = covariance + 1j * A

    # Inverse of the covariance
    c_inverse = ut.compute_inverse(covariance)

    # Log determinant (real for a Hermitian positive-definite covariance)
    _, logdet = jnp.linalg.slogdet(covariance)

    # data term
    data_term = jnp.abs(jnp.einsum("ijk,ikj->i", c_inverse, data))

    # return the likelihood
    return -jnp.sum(jnp.real(logdet) + data_term)


def log_posterior(
    signal_parameters,
    data,
    frequency,
    signal_model,
    response_IJ,
    strain_omega,
    priors,
    response_IJ_V=None,
    signal_model_V=None,
):
    """
    Compute the logarithm of the posterior probability summing log likelihood
    and prior.

    When `response_IJ_V` and `signal_model_V` are provided, circular
    polarisation (Stokes V) is included. The sampled `signal_parameters` are
    then split into an intensity block (the first
    ``len(signal_model.parameter_names)`` entries, passed to `signal_model`) and
    a trailing V-mode block (passed to `signal_model_V`); `priors` must cover
    both blocks. With the V arguments left as ``None`` the behaviour is
    identical to the intensity-only case.

    Parameters:
    -----------
    signal_parameters : Array
        Array containing parameters of the signal model (intensity block
        followed by the optional V-mode block).
    data : Array
        Array containing the observed data.
    frequency : Array
        Array containing frequency bins.
    signal_model : signal_model object
        Object containing the signal model and its derivatives
    response_IJ : Array
        Array containing response function.
    strain_omega : Array
        Array containing strain noise.
    priors : prior object
        Object containing the prior probability density functions.
    response_IJ_V : Array, optional
        Contracted V-mode response, shape (F, N, N). Default is None.
    signal_model_V : signal_model object, optional
        Signal model for the V-mode spectrum. Default is None.

    Returns:
    --------
    float
        Logarithm of the posterior probability.

    """

    # Parameter names, extended with the V-mode block when present
    parameter_names = list(signal_model.parameter_names)
    if response_IJ_V is not None and signal_model_V is not None:
        parameter_names = parameter_names + list(
            signal_model_V.parameter_names
        )

    # Evaluate the log prior
    lp = priors.evaluate_log_priors(
        dict(zip(parameter_names, signal_parameters))
    )

    # If the prior is not finite, return -inf
    if not jnp.isfinite(lp):
        return -jnp.inf

    # Evaluate the intensity signal model on its parameter block
    n_I = len(signal_model.parameter_names)
    signal_value = signal_model.template(frequency, signal_parameters[:n_I])

    # Evaluate the V-mode signal model on the trailing parameter block
    signal_value_V = None
    if response_IJ_V is not None and signal_model_V is not None:
        signal_value_V = signal_model_V.template(
            frequency, signal_parameters[n_I:]
        )

    # Evaluate the log likelihood
    log_lik = log_likelihood(
        data,
        signal_value,
        response_IJ,
        strain_omega,
        response_IJ_V=response_IJ_V,
        signal_value_V=signal_value_V,
    )

    # Return log prior + log likelihood
    return lp + log_lik
