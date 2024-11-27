import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

try:
    get_ipython()
    matplotlib.use('Agg')  # Switch to a non-interactive backend for Jupyter
except NameError:
    matplotlib.use('TkAgg')  # Use GUI backend for terminal or script

def IMF(m):
    """
    Calculate the probabilities of existence using the Kroupa IMF (Kroupa, P. (2001))

    input:
    --------
    m (float or np.array):
        Star mass in Solar Masses [Msun]

    return
    --------
    Probabilities of existence (dimensionless)
    """
    P_m = np.where(
        m < 0.5, 
        0.08**-0.3 * (m / 0.08)**-1.3,
        0.08**-0.3 * (0.5 / 0.08)**-1.3 * (m / 0.5)**-2.3
    )
    return P_m


def TMS(m, time):
    """
    Calculate if the star is on the Main Sequence based on its mass.

    Input:
    --------
    m (float or np.array):
        Star mass in Solar masses [Msun].

    time (float or np.array):
        Time of birth in years [Myr].

    Return:
    --------
    difference (float or np.array):
        Difference between the age of birth and the time on the MS [Myr].
        Negative value can be interpreted as the remaining time on the MS.
    """
    TMS = 10**10 / m**2.5
    return time - TMS

def wd_mass_low(star_mass):
    """
    Calculate WD mass for stars with mass < 8.
    Formula derived from the IFMR in section 8.4 of Kalirai et al. (2008).

    Input:
    --------
    star_mass (float or np.array):
        Initial stellar mass in Solar masses [Msun].

    Return:
    --------
    wd_mass (float or np.array):
        White dwarf mass in Solar masses [Msun].
    """
    return 0.109 * star_mass + 0.394

def wd_mass_mid(star_mass):
    """
    Calculate WD mass for stars with mass between 8 and 13.
    Formula derived from the IFMR in section 8.4 of Kalirai et al. (2008).

    Input:
    --------
    star_mass (float or np.array):
        Initial stellar mass in Solar masses [Msun].

    Return:
    --------
    wd_mass (float or np.array):
        White dwarf mass in Solar masses [Msun].
    """
    return 2.24 + 0.508 * (star_mass - 14.75) + 0.125 * (star_mass - 14.75)**2 + 0.0110 * (star_mass - 14.75)**3

def wd_mass_high(star_mass):
    """
    Calculate WD mass for stars with mass between 12 and 15.
    Formula derived from the IFMR in section 8.4 of Kalirai et al. (2008).

    Input:
    --------
    star_mass (float or np.array):
        Initial stellar mass in Solar masses [Msun].

    Return:
    --------
    wd_mass (float or np.array):
        White dwarf mass in Solar masses [Msun].
    """
    return 0.123 + 0.112 * star_mass

def Neutron_mass_single(m):
    """
    Calculation of the resulting neutron star mass for a given star mass.
    Derived from the IFMR in section 5 of Raithel et al. (2018),
    specifically equations 11, 12, 13, and 14.

    Input:
    --------
    m (float):
        Initial stellar mass in Solar masses [Msun].

    Return:
    --------
    neutron_mass (float):
        Neutron star mass in Solar masses [Msun].
    """
    if (m>=8) and (m<15):
        Mass = 2.24 + 0.508 * (m - 14.75) \
        + 0.125 * (m - 14.75)**2 \
        + 0.0110 * (m - 14.75)**3
    if (m>=15) and (m<17.8):
        Mass = 0.996 + 0.0384*m
    elif (m>17.8) and (m<18.5):
        Mass = -0.020 + 0.10*m
    elif (m>25.2) and (m<27.5):
        Mass = 3232.29 - 409.429*(m - 2.619) + 17.2867*((m - 2.619)**2)- 0.24315*(m - 2.619)**3
    return Mass

def BH_Mass_single(m):
    """
    Calculation of the resulting black hole mass for a given star mass.
    Derived from the IFMR in section 4 of Raithel et al. (2018),
    specifically formulas 1, 2, 3, and 4 with fEJ=0.9.

    Input:
    --------
    m (float):
        Initial stellar mass in Solar masses [Msun].

    Return:
    --------
    bh_mass (float):
        Black hole mass in Solar masses [Msun].
    """
    bh_mass = 0.9 * MBH_core_single(m) + (1 - 0.9) * MBH_all_single(m)
    return bh_mass

def MBH_core_single(m):
    """
    Calculation of the resulting black hole's core mass for a given star mass.
    Derived from the IFMR in section 4 of Raithel et al. (2018).

    Input:
    --------
    m (float):
        Initial stellar mass in Solar masses [Msun].

    Return:
    --------
    core_mass (float):
        Core mass of the black hole in Solar masses [Msun].
    """
    if (m>=15) and (m<=40):
        core_mass = -2.049 + 0.4140*m
    elif (m >= 45):
        core_mass = 5.697 + 7.8598*(10**8) * m**(-4.858)
    return core_mass


def MBH_all_single(m):
    """
    Calculation of the resulting black hole mass for a given star mass.
    Derived from the IFMR in section 4 of Raithel et al. (2018).

    Input:
    --------
    m (float):
        Initial stellar mass in Solar masses [Msun].

    Return:
    --------
    all (float):
        Total black hole mass in Solar masses [Msun].
    """
    
    if (m>15) and (m<=40):
        all = 15.52 - 0.3294*(m-25.97) \
        - 0.02121*(m-25.97)**2 \
        + 0.003120*(m-25.97)**3
    else:
        all = 0
    return all
##################################
def Neutron_mass(m):

    """
    Calculate the resulting neutron star mass for a given star mass.
    Based on the IFMR in section 5 of Raithel et al. (2018),
    using equations 11, 12, 13, and 14.

    Input:
    -------
    m (np.array):
        Star mass in Solar masses [Msun].

    Return:
    --------
    mass (np.array):
        Mass of the neutron star in Solar masses [Msun].
    """
    mass = np.zeros_like(m, dtype=float)
    cond1 = (m >= 15) & (m < 17.8)
    cond2 = (m > 17.8) & (m < 18.5)
    cond3 = (m > 25.2) & (m < 27.5)

    mass[cond1] = 0.996 + 0.0384 * m[cond1]
    mass[cond2] = -0.020 + 0.10 * m[cond2]
    mass[cond3] = (
        3232.29 
        - 409.429 * (m[cond3] - 2.619)
        + 17.2867 * ((m[cond3] - 2.619)**2)
        - 0.24315 * (m[cond3] - 2.619)**3
    )
    return mass

def MBH_core(m):
    """
    Calculate the core mass of a black hole for a given star mass.
    Based on the IFMR in section 4 of Raithel et al. (2018),
    using formulas 1 and 2.

    Input:
    -------
    m (np.array):
        Star mass in Solar masses [Msun].

    Return:
    --------
    core (np.array):
        Core mass of the black hole in Solar masses [Msun].
    """
    core = np.zeros_like(m, dtype=float)
    cond1 = (m >= 15) & (m <= 40)
    cond2 = (m > 45)

    core[cond1] = -2.049 + 0.4140 * m[cond1]
    core[cond2] = 5.697 + 7.8598 * (10**8) * m[cond2]**(-4.858)
    return core

def MBH_all(m):
    """
    Calculate the total mass of a black hole for a given star mass.
    Based on the IFMR in section 4 of Raithel et al. (2018),
    using formulas 3 and 4.

    Input:
    -------
    m (np.array):
        Star mass in Solar masses [Msun].

    Return:
    --------
    all_mass (np.array):
        Total black hole mass in Solar masses [Msun].
    """
    all_mass = np.zeros_like(m, dtype=float)
    cond1 = (m > 15) & (m <= 40)

    all_mass[cond1] = (
        15.52 
        - 0.3294 * (m[cond1] - 25.97)
        - 0.02121 * (m[cond1] - 25.97)**2
        + 0.003120 * (m[cond1] - 25.97)**3
    )
    return all_mass

def BH_Mass(m):
    """
    Calculate the total black hole mass for a given star mass.
    Combines core and total mass contributions using ejection fraction fEJ=0.9.
    Based on the IFMR in section 4 of Raithel et al. (2018).

    Input:
    -------
    m (np.array):
        Star mass in Solar masses [Msun].

    Return:
    --------
    mass (np.array):
        Total mass of the black hole in Solar masses [Msun].
    """
    return 0.9 * MBH_core(m) + (1 - 0.9) * MBH_all(m)

def Mass_random(m):
    """
    Assign a random mass to each star based on a 50% probability.
    With a 50% chance, assign the mass of a black hole; otherwise, assign the mass of a neutron star.

    Input:
    -------
    m (np.array):
        Star mass in Solar masses [Msun].

    Return:
    --------
    result (np.array):
        Random mass assigned to each star, either a black hole mass or a neutron star mass, in Solar masses [Msun].
    """
    tickets = np.random.choice([0, 1], size=m.shape)
    result = np.where(
        tickets == 0, 
        Neutron_mass(m), 
        BH_Mass(m)
    )
    return result

x = np.array([40, 45])
y = np.array([BH_Mass_single(40), BH_Mass_single(45)])


def get_interpolated_values(mass_values):
    """
    Interpolate black hole masses for a given set of star masses within the range [40, 45] Solar masses.
    The interpolation is performed linearly based on predefined data points for star masses of 40 and 45.

    Input:
    -------
    mass_values (np.array):
        Array of star masses in Solar masses [Msun] for which to calculate the interpolated black hole masses.

    Return:
    --------
    interpolated_masses (np.array):
        Interpolated black hole masses corresponding to the input star masses, in Solar masses [Msun].
    """
    return np.interp(mass_values, x, y)

###############################################################################################################################################

def Mass_random_label(m):
    """
    Stocastic function for randomly splitting between black holes and neutron
    stars for the mass range superposition

    input:
    -------
    mass (np.array):
        star mass in Solar masses

    return
    --------
    Label (string):
        Definition of BH or NS for the remanent   

    """
    tickets = np.random.choice([0, 1], size=m.shape)
    label = np.where(
        tickets == 0, 
        'NS', 
        'BH'
    )
    return label



def MC(number, seed,graphs=False):
    """
    Main function to run the Monte Carlo Simulation.
    This function generates a simulated population of stars in the Milky Way,
    computes their final states (e.g., main sequence, white dwarf, neutron star, or black hole),
    and returns a DataFrame with detailed information about each star.

    Input:
    --------
    number (int):
        Number of stars to simulate in the population.

    seed (int):
        Seed for the random number generator to ensure reproducibility.

    graphs (bool):
        If True, the function generates visualizations of the simulation results.

    Return:
    --------
    DataFrame (pandas.DataFrame):
        A DataFrame containing detailed information about the simulated stars, including:
        - Initial masses [Msun].
        - Final states (e.g., MS, WD, NS, BH).
        - Final masses [Msun].
        - Additional parameters such as age of the system and evolutionary stage.

    Units:
    --------
    - Masses are given in Solar masses [Msun].
    - Ages are in millions of years [Myr].

    Notes:
    --------
    This function uses stochastic modeling and employs the Initial Mass Function (IMF)
    to generate a realistic star population. It integrates functions to determine
    the final states and masses of stars, based on literature-referenced Initial-Final Mass Relations (IFMRs):
        - For white dwarfs: Section 8.4 of Kalirai et al. (2008).
        - For neutron stars and black holes: Section 5 and 4 of Raithel et al. (2018), respectively.
    """
    np.random.seed(seed)

    m = np.random.uniform(0.08, 100, number)
    pi = np.random.uniform(0,1,number)
    pIMF = IMF(m) / IMF(0.08)
    Data = pd.DataFrame({'Star mass': m, 'Prob Uniform': pi, 'Prob IMF': pIMF})
    
    Data_filtered = Data[(Data['Prob Uniform'] < Data['Prob IMF'])]
    
    # Time of birth
    N_masked = len(Data_filtered['Star mass'])
    T_Birth = np.random.uniform(0, 10e9, N_masked)

    Data_filtered['Age'] = T_Birth

    Data = pd.DataFrame({'Star mass': Data_filtered['Star mass'], 'Prob existence': Data_filtered['Prob Uniform'],'Age of Birth': Data_filtered['Age']
    }).reset_index(drop=True)

    Time_on_MS = TMS(Data['Star mass'], Data['Age of Birth'])
    Data['Time out MS'] = Time_on_MS

    ##############################################################################################################################################

    cond_interp = (Data['Star mass'] >= 40) & (Data['Star mass'] <= 45)
    interpolated_values = np.full_like(Data['Star mass'], fill_value=np.nan, dtype=float)
    interpolated_values[cond_interp] = get_interpolated_values(Data['Star mass'][cond_interp])



    Mass_final_labels = np.where(Data['Time out MS'] < 0, 'MS', 
                    np.where(Data['Star mass'] < 8, 'WD',
                            np.where((Data['Star mass'] >= 8) & (Data['Star mass'] < 13), 'NS',
                                    np.where((Data['Star mass'] >= 12) & (Data['Star mass'] < 15), 'NS', 
                                                np.where((Data['Star mass'] > 15) & (Data['Star mass'] <= 27.5), Mass_random_label(Data['Star mass'].values), 
                                                        np.where((Data['Star mass']>27.5) & (Data['Star mass']<40), 'BH',
                                                                np.where((Data['Star mass'] >= 40) & (Data['Star mass'] <= 45), 'BH', 
                                                                np.where((Data['Star mass'] >= 45), 'BH', 'MS'
                                                                        ))))))))

    Data['Final Label'] = Mass_final_labels

    Mass_final = np.where(Data['Time out MS'] < 0, Data['Star mass'], 
                          np.where(Data['Star mass'] < 8, wd_mass_low(Data['Star mass']),
                                np.where((Data['Star mass'] >= 8) & (Data['Star mass'] < 13), wd_mass_mid(Data['Star mass']),
                                          np.where((Data['Star mass'] >= 12) & (Data['Star mass'] < 15), wd_mass_high(Data['Star mass']), 
                                                    np.where((Data['Star mass'] > 15) & (Data['Star mass'] <= 27.5) & (Data['Final Label']=="BH"), BH_Mass(Data['Star mass']),
                                                        np.where((Data['Star mass'] > 15) & (Data['Star mass'] <= 27.5) & (Data['Final Label']=="NS"), Neutron_mass(Data['Star mass']),
                                                              np.where((Data['Star mass']>27.5) & (Data['Star mass']<40), BH_Mass(Data['Star mass']),
                                                                    np.where(cond_interp, interpolated_values, 
                                                                        np.where((Data['Star mass'] >= 45), BH_Mass(Data['Star mass']), Data['Star mass']
                                                                                )))))))))





    Data['Final Mass'] = Mass_final
   

    if graphs==True:
        valid_labels = ['MS', 'WD', 'NS', 'BH']
        data_filtered = Data[Data['Final Label'].isin(valid_labels)]

        ordered_labels = ['MS', 'WD', 'NS', 'BH']
        fractions_ordered = (
         data_filtered['Final Label']
           .value_counts(normalize=True)
           .reindex(ordered_labels) * 100
        )

        # Extract data for each category
        M_MS = Data[Data['Final Label'] == 'MS']['Final Mass']
        M_WD = Data[Data['Final Label'] == 'WD']['Final Mass']
        M_NS = Data[Data['Final Label'] == 'NS']['Final Mass']
        M_BH = Data[Data['Final Label'] == 'BH']['Final Mass']

        #   Extract ages for each category
        ages_ms = Data[Data['Final Label'] == 'MS']['Age of Birth']
        ages_wd = Data[Data['Final Label'] == 'WD']['Age of Birth']
        ages_ns = Data[Data['Final Label'] == 'NS']['Age of Birth']
        ages_bh = Data[Data['Final Label'] == 'BH']['Age of Birth']

        masses = {label: Data[Data['Final Label'] == label]['Final Mass'] for label in ordered_labels}
        ages = {label: Data[Data['Final Label'] == label]['Age of Birth'] for label in ordered_labels}

        # Calculate min and max values for mass
        min_max_values = data_filtered.groupby('Final Label')['Final Mass'].agg(['min', 'max'])
        min_max_values = min_max_values.reindex(ordered_labels)
        categories = min_max_values.index
        min_values = min_max_values['min']
        max_values = min_max_values['max']

        extremes = {
    label: {
        "youngest_mass": (Data.loc[Data[(Data['Final Label'] == label) & (Data['Age of Birth'] == ages[label].min())].index, 'Final Mass'].values[0] 
                          if len(Data.loc[Data[(Data['Final Label'] == label) & (Data['Age of Birth'] == ages[label].min())].index, 'Final Mass'].values) > 0 
                          else np.nan),
        "oldest_mass": (Data.loc[Data[(Data['Final Label'] == label) & (Data['Age of Birth'] == ages[label].max())].index, 'Final Mass'].values[0] 
                        if len(Data.loc[Data[(Data['Final Label'] == label) & (Data['Age of Birth'] == ages[label].max())].index, 'Final Mass'].values) > 0 
                        else np.nan),
    }
    for label in ordered_labels
}

    if graphs==True:
        valid_labels = ['MS', 'WD', 'NS', 'BH']
        data_filtered = Data[Data['Final Label'].isin(valid_labels)]

        ordered_labels = ['MS', 'WD', 'NS', 'BH']
        fractions_ordered = (
            data_filtered['Final Label']
           .value_counts(normalize=True)
           .reindex(ordered_labels) * 100
        )

        # Extract data for each category
        M_MS = Data[Data['Final Label'] == 'MS']['Final Mass']
        M_WD = Data[Data['Final Label'] == 'WD']['Final Mass']
        M_NS = Data[Data['Final Label'] == 'NS']['Final Mass']
        M_BH = Data[Data['Final Label'] == 'BH']['Final Mass']

        # Extract ages for each category
        ages_ms = Data[Data['Final Label'] == 'MS']['Age of Birth']
        ages_wd = Data[Data['Final Label'] == 'WD']['Age of Birth']
        ages_ns = Data[Data['Final Label'] == 'NS']['Age of Birth']
        ages_bh = Data[Data['Final Label'] == 'BH']['Age of Birth']

        masses = {label: Data[Data['Final Label'] == label]['Final Mass'] for label in ordered_labels}
        ages = {label: Data[Data['Final Label'] == label]['Age of Birth'] for label in ordered_labels}

        # Calculate min and max values for mass
        min_max_values = data_filtered.groupby('Final Label')['Final Mass'].agg(['min', 'max'])
        min_max_values = min_max_values.reindex(ordered_labels)
        categories = min_max_values.index
        min_values = min_max_values['min']
        max_values = min_max_values['max']

        # Identify youngest and oldest masses for each category
        extremes = {
            label: {
                "youngest_mass": (Data.loc[Data[(Data['Final Label'] == label) & (Data['Age of Birth'] == ages[label].min())].index, 'Final Mass'].values[0] 
                                  if len(Data.loc[Data[(Data['Final Label'] == label) & (Data['Age of Birth'] == ages[label].min())].index, 'Final Mass'].values) > 0 
                                  else np.nan),
                "oldest_mass": (Data.loc[Data[(Data['Final Label'] == label) & (Data['Age of Birth'] == ages[label].max())].index, 'Final Mass'].values[0] 
                                if len(Data.loc[Data[(Data['Final Label'] == label) & (Data['Age of Birth'] == ages[label].max())].index, 'Final Mass'].values) > 0 
                                else np.nan),
            }
            for label in ordered_labels
        }

        categories = list(extremes.keys())
        youngest_masses = [extremes[label]['youngest_mass'] for label in categories]
        oldest_masses = [extremes[label]['oldest_mass'] for label in categories]


        # Create bins for histograms
        bins_mass = np.linspace(min(M_MS.min(), M_WD.min(), M_NS.min(), M_BH.min()),
                                max(M_MS.max(), M_WD.max(), M_NS.max(), M_BH.max()), 100)
        bins_age = np.linspace(min(ages_ms.min(), ages_wd.min(), ages_ns.min(), ages_bh.min()),
                                 max(ages_ms.max(), ages_wd.max(), ages_ns.max(), ages_bh.max()), 100)

        strong_colors = {
            "MS": "blue",
            "WD": "orange",
            "NS": "green",
            "BH": "red"
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)

        # First subplot: Normalized Mass Histogram
        hist_ms, _ = np.histogram(M_MS, bins=bins_mass, density=True)
        hist_wd, _ = np.histogram(M_WD, bins=bins_mass, density=True)
        hist_ns, _ = np.histogram(M_NS, bins=bins_mass, density=True)
        hist_bh, _ = np.histogram(M_BH, bins=bins_mass, density=True)

        # Normalize histograms, using np.where to avoid division by zero
        hist_ms_normalized = np.where(max(hist_ms) > 0, hist_ms / max(hist_ms), hist_ms)
        hist_wd_normalized = np.where(max(hist_wd) > 0, hist_wd / max(hist_wd), hist_wd)
        hist_ns_normalized = np.where(max(hist_ns) > 0, hist_ns / max(hist_ns), hist_ns)
        hist_bh_normalized = np.where(max(hist_bh) > 0, hist_bh / max(hist_bh), hist_bh)

        # Plot histograms
        axes[0, 0].bar(bins_mass[:-1], hist_ms_normalized, width=np.diff(bins_mass), 
                       color=strong_colors["MS"], edgecolor='black', alpha=0.8, label="#MS = "+str(len(M_MS)))
        axes[0, 0].bar(bins_mass[:-1], hist_wd_normalized, width=np.diff(bins_mass), 
                     color=strong_colors["WD"], edgecolor='black', alpha=0.8, label="#WD = "+str(len(M_WD)))
        axes[0, 0].bar(bins_mass[:-1], hist_ns_normalized, width=np.diff(bins_mass), 
                    color=strong_colors["NS"], edgecolor='black', alpha=0.8, label="#NS = "+str(len(M_NS)))
        axes[0, 0].bar(bins_mass[:-1], hist_bh_normalized, width=np.diff(bins_mass), 
                       color=strong_colors["BH"], edgecolor='black', alpha=0.8, label="#BH = "+str(len(M_BH)))

        axes[0, 0].set_xlabel("Mass [Msun]")
        axes[0, 0].set_ylabel("Normalized Histogram")
        axes[0, 0].legend()

        # Second subplot: Normalized Age Histogram
        hist_ages_ms, _ = np.histogram(ages_ms, bins=bins_age, density=True)
        hist_ages_wd, _ = np.histogram(ages_wd, bins=bins_age, density=True)
        hist_ages_ns, _ = np.histogram(ages_ns, bins=bins_age, density=True)
        hist_ages_bh, _ = np.histogram(ages_bh, bins=bins_age, density=True)

        # Normalize age histograms, using np.where to avoid division by zero
        hist_ages_ms_normalized = np.where(max(hist_ages_ms) > 0, hist_ages_ms / max(hist_ages_ms), hist_ages_ms)
        hist_ages_wd_normalized = np.where(max(hist_ages_wd) > 0, hist_ages_wd / max(hist_ages_wd), hist_ages_wd)
        hist_ages_ns_normalized = np.where(max(hist_ages_ns) > 0, hist_ages_ns / max(hist_ages_ns), hist_ages_ns)
        hist_ages_bh_normalized = np.where(max(hist_ages_bh) > 0, hist_ages_bh / max(hist_ages_bh), hist_ages_bh)

        # Plot histograms for ages
        axes[0, 1].bar(bins_age[:-1], hist_ages_ms_normalized, width=np.diff(bins_age), 
                       color=strong_colors["MS"], edgecolor='black', alpha=0.8, label="#MS = "+str(len(M_MS)))
        axes[0, 1].bar(bins_age[:-1], hist_ages_wd_normalized, width=np.diff(bins_age), 
                       color=strong_colors["WD"], edgecolor='black', alpha=0.8, label="#WD = "+str(len(M_WD)))
        axes[0, 1].bar(bins_age[:-1], hist_ages_ns_normalized, width=np.diff(bins_age), 
                       color=strong_colors["NS"], edgecolor='black', alpha=0.8, label="#NS = "+str(len(M_NS)))
        axes[0, 1].bar(bins_age[:-1], hist_ages_bh_normalized, width=np.diff(bins_age), 
                       color=strong_colors["BH"], edgecolor='black', alpha=0.8, label="#BH = "+str(len(M_BH)))

        axes[0, 1].set_xlabel("Total Age [Myr]")
        axes[0, 1].set_ylabel("Normalized Histogram")
        axes[0, 1].legend()


        # === Bottom-left: Youngest and Oldest Mass by Category ===
        x = np.arange(len(categories))
        width = 0.4
        axes[1, 0].bar(
           x - width / 2, youngest_masses, width,
           label='Youngest', color='blue', alpha=0.8
        )
        axes[1, 0].bar(
           x + width / 2, oldest_masses, width,
           label='Oldest', color='red', alpha=0.8
        )
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].set_ylabel("Mass [Msun]")
        axes[1, 0].set_xlabel("Object Type")
        axes[1, 0].legend()

        # Add value labels above the bars
        for i, (y, o) in enumerate(zip(youngest_masses, oldest_masses)):
            axes[1, 0].text(i - width / 2, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9)
            axes[1, 0].text(i + width / 2, o, f"{o:.2f}", ha='center', va='bottom', fontsize=9)


        # === Fourth subplot: Fraction of Objects by Category ===
        axes[1, 1].bar(fractions_ordered.index, fractions_ordered, color=[strong_colors[label] for label in fractions_ordered.index])
        axes[1, 1].set_xlabel("Object Type")
        axes[1, 1].set_ylabel("Fraction (%)")
        for i, val in enumerate(fractions_ordered):
            axes[1, 1].text(i, val + 1, f"{val:.1f}%", ha='center', fontsize=10)

        for ax in axes.ravel():
            ax.grid(False)

        plt.tight_layout()
        plt.show()
    return Data

#Comment the following lines if running in jupyter
if __name__ == '__main__':
    number = input('number: ')
    seed = input('seed: ')
    graphs= input('graphs (True/False): ')
    if graphs == 'True':
        graphs = True
    else:
        graphs = False
    MC(int(number), int(seed), graphs)


