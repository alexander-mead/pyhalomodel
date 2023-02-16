from math import pi

# Physical constants
G = 6.6743e-11 # Newton constant [m^3 kg^-1 s^-2]

# Astronomy
Sun_mass = 1.9884e30 # Mass of the Sun [kg]
Mpc = 3.0857e16*1e6  # Mpc [m]

# Cosmology
H0 = 100. # Hubble parameter today in h [km/s/Mpc]
G_cosmological = G*Sun_mass/(Mpc*1e3**2) # Gravitational constant [(Msun/h)^-1 (km/s)^2 (Mpc/h)] ~4.301e-9 (1e3**2 m -> km)
rho_critical = 3.*H0**2/(8.*pi*G_cosmological) # Critical density [(Msun/h) (Mpc/h)^-3] ~2.775e11 (Msun/h)/(Mpc/h)^3
neutrino_constant = 93.1 # Neutrino mass required to close universe [eV]