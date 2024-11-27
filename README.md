This repository contains a Python implementation of a Monte Carlo simulation to study stellar evolution in the Milky Way. 
The code generates synthetic star populations, calculates their final states (main sequence stars, white dwarfs, neutron stars, black holes), and visualizes key insights.

Features
Monte Carlo Simulation:

Generates stars based on the Kroupa Initial Mass Function (IMF).
Assigns stellar remnants using Initial-Final Mass Relationships (IFMR).
Visualization:

Normalized histograms for masses and ages.
Bar charts showing stellar fractions and youngest/oldest masses.
Customizable:

Number of stars (number).
Random seed (seed).
Optional visual outputs (graphs).
Requirements:
python 3.7

result = MC(number=10000, seed=42, graphs=True)
Visualizations:

Mass and age histograms.
Stellar fraction bar chart.
Youngest and oldest stellar masses.
Key Functions
MC(number, seed, graphs): Main simulation function.
IMF(m): Calculates star probabilities using Kroupa IMF (Kroupa, 2001).
BH_Mass(m) / Neutron_mass(m): Calculates black hole/neutron star masses (Raithel et al., 2018).
wd_mass(m): Calculates white dwarf masses (Kalirai et al., 2008).