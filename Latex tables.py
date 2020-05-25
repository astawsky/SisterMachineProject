import pickle
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.linear_model import LinearRegression


def main():
    # Import the objects we want
    pickle_in = open("NewSisterCellClass_Population.pickle", "rb")
    Population = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Sister.pickle", "rb")
    Sister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Nonsister.pickle", "rb")
    Nonsister = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("NewSisterCellClass_Control.pickle", "rb")
    Control = pickle.load(pickle_in)
    pickle_in.close()

    sym_vars = [r'\tau-\overline{\tau}', r'\ln(\frac{x(0)}{\overline{x}})', r'\ln(\frac{x(\tau)}{\overline{x}})', r'\alpha-\overline{\alpha}', r'\phi-\overline{\phi}', r'f-\overline{f}']
    combos = list(itertools.combinations(sym_vars, 2))

    indices = [r'$\rho('+combo[0]+', \, '+combo[1]+')$' for combo in combos]

    table = pd.DataFrame(columns=['Global', 'Trap', 'Traj'], index=indices)

    var_combos = list(itertools.combinations(Population._variable_names, 2))

    # same cell

    mom = Population.mother_dfs[0].copy()
    daug = Population.daughter_dfs[0].copy()

    pop_stats = Population.pop_stats
    pop_stats['length_birth'].loc['mean'] = np.log(pop_stats['length_birth'].loc['mean'])

    mom['length_birth'] = np.log(mom['length_birth'])
    daug['length_birth'] = np.log(daug['length_birth'])
    mom['trap_avg_length_birth'] = np.log(mom['trap_avg_length_birth'])
    daug['trap_avg_length_birth'] = np.log(daug['trap_avg_length_birth'])
    mom['traj_avg_length_birth'] = np.log(mom['traj_avg_length_birth'])
    daug['traj_avg_length_birth'] = np.log(daug['traj_avg_length_birth'])

    mom_global = mom.copy()
    mom_trap = mom.copy()
    mom_traj = mom.copy()

    # Global
    for var in Population._variable_names:
        mom_global[var] = mom_global[var] - pop_stats[var].loc['mean']

    # Trap
    for var in Population._variable_names:
        mom_trap[var] = mom_trap[var] - mom_trap['trap_avg_' + var]

    # Traj
    for var in Population._variable_names:
        mom_traj[var] = mom_traj[var] - mom_trap['traj_avg_' + var]

    for var_combo, index in zip(var_combos, indices):
        table['Global'].loc[index] = round(stats.pearsonr(mom_global[var_combo[0]], mom_global[var_combo[1]])[0], 3)
        table['Trap'].loc[index] = round(stats.pearsonr(mom_trap[var_combo[0]], mom_trap[var_combo[1]])[0], 3)
        table['Traj'].loc[index] = round(stats.pearsonr(mom_traj[var_combo[0]], mom_traj[var_combo[1]])[0], 3)

    print(table.index)
    print(table)
    latex_table = table.to_latex(index=True, index_names=True, escape=False, multirow=True)
    print(latex_table)#
    with open("Output.txt", "w") as text_file:
        print(f"Purchase Amount: {latex_table}", file=text_file)


if __name__ == '__main__':
    main()
