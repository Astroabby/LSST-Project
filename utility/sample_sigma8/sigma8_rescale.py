from numpy import log, pi
from cosmosis.datablock import names as section_names

cosmo = section_names.cosmological_parameters
matter_powspec = section_names.matter_power_lin


def setup(options):
    return 0


def execute(block, config):
    # Get parameters from sampler and CAMB output
    sigma8_input = block[cosmo, 'sigma8_input']
    sigma8_camb = block[cosmo, 'sigma_8']
    A_s = block[cosmo, 'A_s']
    P_k = block[matter_powspec, 'P_k']

    zmin = block[matter_powspec, 'z'].min()
    if zmin != 0.0:
        raise ValueError(
            "You need to set zmin=0 in CAMB to use the sigma8_rescale module."
        )

    # Calculate rescale factor
    r = (sigma8_input**2) / (sigma8_camb**2)

    # Rescale A_s and matter power spectrum
    A_s *= r
    P_k *= r

    # Save back to block
    block[cosmo, 'A_s'] = A_s
    block[matter_powspec, 'P_k'] = P_k
    block[cosmo, 'sigma_8'] = sigma8_input

    return 0


def cleanup(config):
    return 0






