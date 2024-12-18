This repository contains a Python implementation of a Monte Carlo simulation to study stellar evolution in the Milky Way. 
The code generates synthetic star populations, calculates their final states (main sequence stars, white dwarfs, neutron stars, black holes), and visualizes key insights.


# Monte Carlo Simulation:

Generates stars based on the Kroupa Initial Mass Function (IMF).
Assigns stellar remnants using Initial-Final Mass Relationships (IFMR).

# Customizable:

Number of stars (number).

Random seed (seed).

Optional visual outputs (graphs).


# Requirements:

Python 3.7

libraries: numpy, pandas, matplotlib


# Key Functions:

(if running in jupyter comment the lines below line 673)

MC(number, seed, graphs): Main simulation function.

IMF(m): Calculates star probabilities using Kroupa IMF (Kroupa, 2001).

BH_Mass(m) / Neutron_mass(m): Calculates black hole/neutron star masses (Raithel et al., 2018).

wd_mass(m): Calculates white dwarf masses (Kalirai et al., 2008).


# Implementation example

a) Terminal: 

python3 MC_AM.py


b) Jupyter: 

import MC_AM as MC_AM

importlib.reload(MC_AM)

Data = MC_AM.MC(int(1e7), 50, True)


# Visualizations:

Mass and age histograms.
Stellar fraction bar chart.
Youngest and oldest stellar masses.

