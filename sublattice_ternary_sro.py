import argparse 
import re
import os 
from os import listdir

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import ase
from ase.io import read


ref_atoms = ase.io.read("POSCAR_Ni3Al")

def load_poscar(composition, nsolutes):
    """
    Loads all POSCAR files for all distributions of a given alloy composition into DataFrames
    :param str composition: Path to MC files for alloy to be analyzed
    :param int nsolutes: Number of solutes in alloy
    :return: List of DataFrames containing POSCARS for each solute distribution
    Usage: load_poscar("./11_NiAl_Ti", 8)
    """
    run_names = re.compile("Run_\d+")
    sub_names = re.compile("\d{2}")
    sub_dfs = []
    natoms = [8, 27, 44]
    natoms_path = f'{(natoms.index(nsolutes)+1):02d}_{nsolutes}_atoms'
    for substitution in np.sort(list(filter(sub_names.match, listdir(f"{composition}/{natoms_path}")))):
        run_id, poscar_files, empty_files = [], [], []
        for run in list(filter(run_names.match, listdir(f"{composition}/{natoms_path}/{substitution}/01_Structures"))):
            for poscar in listdir(f"{composition}/{natoms_path}/{substitution}/01_Structures/{run}"):
                path = f"{composition}/{natoms_path}/{substitution}/01_Structures/{run}/{poscar}"
                run_num = re.search("[1-9]+\d*", run).group(0)
                poscar_num = re.search("\d+", poscar).group(0)
                try:
                    poscar_files.append(ase.io.read(path, format="vasp")) 
                    run_id.append(f"{run_num}-{poscar_num}")
                except:
                    empty_files.append(f"{run_num}-{poscar_num}")
                    pass
        sub_dfs.append(pd.DataFrame({"Run ID":run_id, "POSCAR":poscar_files}).sort_values("Run ID").reset_index().drop(columns="index"))
        print(empty_files)
        if not os.path.exists('poscar_dfs'):
            os.mkdir('poscar_dfs')
        sub_dfs[-1].to_csv(os.path.join('poscar_dfs', f'{composition}_{nsolutes}_{substitution}_POSCARs.csv'), index=False)
    return sub_dfs

def plot_wc_evolution(wc_df, composition, nsolutes, distrib, distrib_count, ref_distrib):
    """
    Generates plot of Warren-Cowley parameter evolution for a specific alloy composition and distribution over time and saves to png
    :param df wc_df: DataFrame of all Warren-Cowley parameters for composition/distribution
    :param str composition: Chemical composition of alloy to be analyzed
    :param int nsolutes: Number of solute atoms in alloy
    :param str distrib: Distribution of solute species in alloy (eg 00, 01, etc)
    :param Series distrib_count: Series containing counts of species in starting MC structure (used to calculate dnumeric distribution of solutes: x% in Al, y% in Ni)
    :param Atoms ref_distrib: Series containing counts of species in pure NiAl structure
    :return: None
    Usage: plot_wc_evolution(process_wc(load_poscar("./11_NiAl_Ti", 8)[0], "NiAl_Ti", 8, "00", 1), "NiAl_Ti", 8, "00", pd.Series(poscar_df["POSCAR"][0].get_chemical_symbols()).value_counts(), 1, pd.Series(ref_atoms.get_chemical_symbols()).value_counts())
    """
    plt.rcParams.update({'figure.figsize':(15, 7),       
                     'font.size':14,               
                     'font.family':'sans-serif',   
                     'mathtext.fontset':'cm',      
                     'mathtext.rm':'serif',        
                    })
    if not os.path.exists('sublattice_fig'):
        os.mkdir('sublattice_fig')
    for sublattice in ["Ni Lattice", "Al Lattice"]:
        df = wc_df.loc[:, wc_df.columns.str.contains(sublattice)]
        df.columns = df.columns.str[-5:]
        fig, ax = plt.subplots(dpi=1200)
        for col in df.columns:
            ax.plot(df.index.values, df[col], label=col)
        plt.xlabel("MC Timestep")
        plt.ylabel(f"WC Parameter (1-NN)")
        alloy = composition[3:]
        plt.title(f"{sublattice[:2]} sublattice WC parameters for {alloy}_{nsolutes} with {int(100*(ref_distrib['Ni']-distrib_count['Ni'])/nsolutes)}% Ni, {int(100*(ref_distrib['Al']-distrib_count['Al'])/nsolutes)}% Al starting distribution (1-NN)")
        plt.legend(loc="upper right")
        plt.xlim([0, df.shape[0]*1.2])
        fig.savefig(os.path.join('sublattice_data', f'{composition}_{nsolutes}_{distrib}_{sublattice}.png'), dpi=300, bbox_inches='tight', transparent=False)

def nearest_neighbors(atoms, min_threshold, max_threshold):
    """
    Outputs 2D array of nearest neighbor species for each atom in an Atoms object
    :param Atoms atoms: Atoms object to be analyzed
    :param int min_threshold: Nth closest nearest neighbor to be considered
    :param int max_threshold: Nth farthest nearest neighbor to be considered
    :return: 2D array of nearest neighbor species for each atom
    Usage: nearest_neighbors(load_poscar("./11_NiAl_Ti", 8)[0]["POSCAR"][0], 1, 12)
    """
    species = np.array(atoms.get_chemical_symbols())
    all_distances = np.round(atoms.get_all_distances(mic=True), 1)
    idx = np.argsort(all_distances, axis=1)[:, 1:]
    return species[idx[:, min_threshold-1:max_threshold]]

def warren_cowley(j, neighbors_arr, c_j):
    """
    Calculates Warren-Cowley parameters with respect to species `j`
    :param str j: Species of neighbor atom
    :param arr neighbors_arr: Array containing species of nearest neighboring atoms
    :param float c_j: Concentration of neighbor atom `j` across structure
    :return: Warren-Cowley parameter for all atoms in `neighbors_arr` relative to species `j`
    Usage: warren_cowley("Ti", nearest_neighbors(load_poscar("./11_NiAl_Ti", 8)[0]["POSCAR"][0], 1, 12), 0.5)
    """
    return 1 - np.mean(neighbors_arr == j)/c_j

def wc_parameters(atoms, ref_atoms, solute):
    """
    Generates DataFrame containing Warren-Cowley parameters for all possible species pairs in an Atoms object
    :param Atoms atoms: Atoms object to be analyzed
    :param int min_threshold: Nth closest nearest neighbor to be considered
    :param int max_threshold: Nth farthest nearest neighbor to be considered
    :param Atoms ref_atoms: Pure Ni3Al Atoms object used to reference Ni/Al lattice sites
    :param str solute: Element acting as alloy solute
    :return: DataFrame of Warren-Crowley parameters for all species in `atoms`
    Usage: wc_parameters(load_poscar("./11_NiAl_Ti", 8)[0]["POSCAR"][0], 1, 12, ref_atoms, "Ti")
    """
    ref_sublattice = np.array(ref_atoms.get_chemical_symbols())
    ref_matrix = np.reshape(np.repeat(ref_atoms.get_positions(), 432), (432, 3, 432))
    atoms_matrix = atoms.get_positions()[:, :, np.newaxis].T
    atoms_sites = np.argmin(np.linalg.norm(atoms_matrix - ref_matrix, axis=1)[:, np.newaxis], axis=0).flatten()
    atoms_sublattice = ref_sublattice[atoms_sites]
    ni_lattice = atoms[atoms_sublattice == "Ni"]
    al_lattice = atoms[atoms_sublattice == "Al"]
    all_species = np.unique(atoms.get_chemical_symbols())
    wc_outputs = []
    subs = {"Ni":[ni_lattice, (1, 8)], "Al":[al_lattice, (1, 6)]}
    for sublattice in subs:
        sublattice_obj = subs[sublattice][0]
        sublattice_species = sublattice_obj.get_chemical_symbols()
        min_threshold, max_threshold = subs[sublattice][1]
        neighbors = nearest_neighbors(sublattice_obj, min_threshold, max_threshold)
        wc_values = np.zeros((len(neighbors), len(all_species)))
        for i, species in enumerate(all_species):
            c_j = np.mean(np.array(sublattice_species) == species)
            wc_values[:, i] = np.apply_along_axis(lambda arr: warren_cowley(species, arr, c_j), 1, neighbors)
        wc_df = pd.DataFrame(wc_values, columns = all_species)
        wc_df["Species"] = sublattice_species
        aggregated = wc_df.groupby("Species").mean().round(3)
        for missing in all_species[~np.isin(all_species, aggregated.index)]:
            aggregated[missing] = [np.nan]*len(aggregated)
            aggregated.loc[missing, :] = [np.nan]*len(aggregated.columns)
        wc_outputs.append(aggregated.sort_index())
    return wc_outputs

def process_wc(poscar_df, composition, nsolutes, distrib, ref_atoms, plot=True):
    """
    Generates DataFrame containing all 1-NN Warren-Cowley parameters for all Atoms objects in a given MC run and saves to csv
    :param df poscar_df: DataFrame of POSCAR files for each MC run
    :param str composition: Chemical composition of alloy to be analyzed
    :param int nsolutes: Number of solute atoms in each Atoms object of `poscar_df`
    :param str distrib: Distribution of solute species in alloy (eg 00, 01, etc)
    :param Atoms ref_atoms: Atoms object used to reference Ni/Al lattice sites
    :param bool plot: Flag to plot evolution of WC values
    :return: DataFrame of all WC parameters across all time-steps of MC run
    Usage: process_wc(load_poscar("./11_NiAl_Ti", 8)[0], "NiAl_Ti", 8, "00", 1, ase.io.read("POSCAR_Ni3Al"))
    """
    species = np.unique(poscar_df["POSCAR"][0].get_chemical_symbols())
    upper_tri_indices = np.triu_indices(len(species))
    wcs = np.zeros((len(poscar_df), 2*len(upper_tri_indices[0])))
    for i in range(0, len(poscar_df)):
        ni_al_wc = wc_parameters(poscar_df["POSCAR"][i], ref_atoms, species[(species!="Al")&(species!="Ni")][0])
        ni_wc, al_wc = np.array(ni_al_wc[0]), np.array(ni_al_wc[1])
        wcs[i, :int(wcs.shape[1]/2)] = ni_wc[upper_tri_indices]
        wcs[i, int(wcs.shape[1]/2):] = al_wc[upper_tri_indices]
    species_pair_df = ni_al_wc[0] #Arbitrarily grab wc df for column namings
    ni_cols = [f"Ni Lattice {species_pair_df.index[pair[0]]}-{species_pair_df.columns[pair[1]]}" for pair in zip(upper_tri_indices[0], upper_tri_indices[1])]
    al_cols = [f"Al Lattice {species_pair_df.index[pair[0]]}-{species_pair_df.columns[pair[1]]}" for pair in zip(upper_tri_indices[0], upper_tri_indices[1])]
    wc_vals = pd.DataFrame(columns=ni_cols + al_cols, data=wcs)
    output = poscar_df.merge(wc_vals, left_index=True, right_index=True)
    if not os.path.exists('sublattice_data'):
        os.mkdir('sublattice_data')
    output.to_csv(os.path.join('sublattice_data', f'{composition}_{nsolutes}_{distrib}.csv'), index=False)
    if plot:
        ref_distrib = pd.Series(ref_atoms.get_chemical_symbols()).value_counts()
        distrib_count = pd.Series(poscar_df["POSCAR"][0].get_chemical_symbols()).value_counts()
        plot_wc_evolution(output, composition, nsolutes, distrib, distrib_count, ref_distrib)
    return output
    

def main():
    ''' 
    Usage:   python ternary_sro.py . -n 8 -s Ti -d 00 --plot
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('root',
                        help='the root directory',
                        type=str)
    parser.add_argument('-n', '--nsolutes',
                        help='the number of total solute atoms',
                        type=int,
                        required=True)
    parser.add_argument('-s', '--solutes',
                        help='the solute species',
                        nargs='+',
                        type=str,
                        required=True)
    parser.add_argument('-d', '--distrib',
                        help='the distribution of solutes',
                        type=str,
                        required=True)
    parser.add_argument('--plot',
                        help='plot the WC parameters vs. MCS',
                        action='store_true')

    args = parser.parse_args()
    root = args.root
    nsolutes = args.nsolutes
    solutes = args.solutes
    distrib = args.distrib
    plot_flag = args.plot

    ternary = ['Ti', 'Nb', 'Co', 'Fe']
    main_folder = f'{ternary.index(solutes[0])+11}_NiAl_{solutes[0]}'
    folder = os.path.join(root, main_folder)
    print(f'Accessing {folder} with {nsolutes} atoms of {solutes} distributed as {distrib}.')
    print(f'Calculating the WC SRO parameters for the 1-NN shell.')

    data  = load_poscar(folder, nsolutes) 
    print(f'Finished loading POSCARs for {folder}.')

    output = process_wc(data[int(distrib)], main_folder, nsolutes, distrib, ref_atoms, plot_flag)
    print(f'Finished calculating WC params for {main_folder}.')    

if __name__ == '__main__':
    main()



