import time
import numpy as np
import pyccl as ccl

# Definir valores de nk y kmax manualmente
nk = 1000
kmax = 3  # En unidades de Mpc^-1
k = np.logspace(-5, np.log10(kmax), nk)
z_vals = np.linspace(0, 2, 250)  # Igual que CosmoSIS: 0.0, 0.5, 1.0, 1.5, 2.0
a_vals = 1 / (1 + z_vals)

# Definir los parámetros cosmológicos base
# Opción 1: HALOFIT
#cosmo_halofit = ccl.Cosmology(Omega_c=0.2663,
 #   Omega_b=0.049,
  #  h=0.6736,
   # n_s=0.9649,
    #sigma8=0.7933395884481901,
    #transfer_function='boltzmann_camb', matter_power_spectrum="camb",
     #                         extra_parameters={"camb": {"halofit_version": "takahashi"}})

#start = time.time()
#pk_halofit_lin = ccl.linear_matter_power(cosmo_halofit, k, a)  # P(k) lineal
#pk_halofit_nl = ccl.nonlin_matter_power(cosmo_halofit, k, a)  # P(k) no lineal
#time_halofit = time.time() - start

#np.savetxt("matter_power_lin_ccl.txt", np.column_stack([k, pk_halofit_lin]), header="k [1/Mpc]  P_lin(k) [(Mpc/h)^3]")
#np.savetxt("matter_power_nl_ccl.txt", np.column_stack([k, pk_halofit_nl]), header="k [1/Mpc]  P_nl(k) [(Mpc/h)^3]")

#print(f"HALOFIT computing time: {time_halofit:.3f} s")

# Opción 2: HMCode (Mead 2020 Feedback)
cosmo_hmcode = ccl.Cosmology(Omega_c=0.2663,
    Omega_b=0.049,
    h=0.6736,
    n_s=0.9649,
    sigma8=0.7933395884481901,
    transfer_function='boltzmann_camb', matter_power_spectrum="camb", extra_parameters={"camb": {"halofit_version": "mead2020_feedback", "HMCode_logT_AGN": 7.8}})
# Crear matriz vacía para P(k, z)
pk_matrix_lin = []
pk_matrix_nl = []

start = time.time()
for a in a_vals:
   pk_matrix_lin.append(ccl.linear_matter_power(cosmo_hmcode, k, a))
   pk_matrix_nl.append(ccl.nonlin_matter_power(cosmo_hmcode, k, a))
time_hmcode = time.time() - start

pk_matrix_lin = np.array(pk_matrix_lin)  # shape (nz, nk)
pk_matrix_nl = np.array(pk_matrix_nl)

# Guardar los archivos como en CosmoSIS
np.savetxt("z.txt", z_vals, header="z")
np.savetxt("k_h.txt", k, header="k [h/Mpc]")
np.savetxt("p_k_lin.txt", pk_matrix_lin, header="P_lin(k,z) [(Mpc/h)^3]; filas=z, columnas=k")
np.savetxt("p_k_nl.txt", pk_matrix_nl, header="P_nl(k,z) [(Mpc/h)^3]; filas=z, columnas=k")


print(f"HMCode computing time: {time_hmcode:.3f} s")

