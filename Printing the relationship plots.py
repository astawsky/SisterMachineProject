import pickle
import pandas as pd
import numpy as np


def concat_the_AB_traces_for_symmetry(df_A, df_B):
    print(len(df_A.copy()), len(df_B.copy()))
    both_A = pd.concat([df_A.copy(), df_B.copy()], axis=0)
    both_B = pd.concat([df_B.copy(), df_A.copy()], axis=0)
    print(len(both_A), len(both_B))
    return both_A, both_B


def with_different_averages():
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

    Population.plot_same_cell_correlations(df=mom_global, variables=Population._variable_names,
                                           labels=Population._variable_symbols.loc['without subscript, normalized lengths'])

    # Trap
    for var in Population._variable_names:
        mom_trap[var] = mom_trap[var] - mom_trap['trap_avg_'+var]

    Population.plot_same_cell_correlations(df=mom_trap, variables=Population._variable_names,
                                           labels=Population._variable_symbols.loc['without subscript, normalized lengths'])

    # Traj
    for var in Population._variable_names:
        mom_traj[var] = mom_traj[var] - mom_trap['traj_avg_'+var]

    Population.plot_same_cell_correlations(df=mom_traj, variables=Population._variable_names,
                                           labels=Population._variable_symbols.loc['without subscript, normalized lengths'])


def main():

    """
    Here we will plot/save the population level same-cell and (great_/grand_)mother_(great_/grand_)daughter correlations.
    Also we plot/save the dataset dependent sister and (first_/second_)cousin correlations.
    """

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

    # mother_df = Population.mother_df.copy()
    # print(mother_df['generationtime'].isnull().values.any())
    # mother_df['generationtime'] = np.log(mother_df['generationtime'])
    # print(mother_df['generationtime'].isnull().values.any())
    # print(mother_df['growth_rate'].isnull().values.any())
    # mother_df['growth_rate'] = np.log(mother_df['growth_rate'])
    # print(np.argwhere(np.isnan(mother_df['growth_rate'])))
    # print(mother_df['growth_rate'].isnull().values.sum())
    # daughter_df = Population.daughter_df.copy()
    # daughter_df['generationtime'] = np.log(daughter_df['generationtime'])
    # daughter_df['growth_rate'] = np.log(daughter_df['growth_rate'])
    # print(len(mother_df), len(daughter_df))
    #
    # mother_df = mother_df.replace([np.inf, -np.inf], np.nan).dropna()
    # daughter_df = daughter_df.replace([np.inf, -np.inf], np.nan).dropna()
    # print(len(mother_df), len(daughter_df))
    # exit()
    # print(daughter_df.isnull().values.sum())
    # print(mother_df.isnull().values.sum())
    # for var in mother_df.columns:
    #     print(var, np.argwhere(np.isnan(mother_df[var])))
    # for var in daughter_df.columns:
    #     print(var, np.argwhere(np.isnan(daughter_df[var])))
    # exit()

    mother_df = Population.mother_dfs[0]
    daughter_df = Population.daughter_dfs[0]

    # # Same-cell
    Population.plot_same_cell_correlations(df=mother_df, variables=Population._variable_names,
                                           labels=Population._variable_symbols.loc['without subscript, normalized lengths'])

    # Mother-Daughter
    Population.plot_relationship_correlations(df_1=mother_df, df_2=daughter_df, df_1_variables=Population._variable_names,
                                    df_2_variables=Population._variable_names, x_labels=Population._variable_symbols.loc['_n, normalized lengths'],
                                    y_labels=Population._variable_symbols.loc['_{n+1}, normalized lengths'])

    # Sister-Sister for S/N/C datasets
    both_A, both_B = concat_the_AB_traces_for_symmetry(df_A=Sister.A_intra_gen_bacteria[0], df_B=Sister.B_intra_gen_bacteria[0])
    Sister.plot_relationship_correlations(df_1=both_A, df_2=both_B, df_1_variables=Sister._variable_names,
                                              df_2_variables=Sister._variable_names,
                                              x_labels=Sister._variable_symbols.loc['_B, normalized lengths'],
                                              y_labels=Sister._variable_symbols.loc['_A, normalized lengths'])

    both_A, both_B = concat_the_AB_traces_for_symmetry(df_A=Nonsister.A_intra_gen_bacteria[0], df_B=Nonsister.B_intra_gen_bacteria[0])
    Nonsister.plot_relationship_correlations(df_1=both_A, df_2=both_B, df_1_variables=Nonsister._variable_names,
                                          df_2_variables=Nonsister._variable_names,
                                          x_labels=Nonsister._variable_symbols.loc['_B, normalized lengths'],
                                          y_labels=Nonsister._variable_symbols.loc['_A, normalized lengths'])

    both_A, both_B = concat_the_AB_traces_for_symmetry(df_A=Control.A_intra_gen_bacteria[0], df_B=Control.B_intra_gen_bacteria[0])
    Control.plot_relationship_correlations(df_1=both_A, df_2=both_B, df_1_variables=Control._variable_names,
                                          df_2_variables=Control._variable_names,
                                          x_labels=Control._variable_symbols.loc['_B, normalized lengths'],
                                          y_labels=Control._variable_symbols.loc['_A, normalized lengths'])

    # # Cousin-Cousin for S/N/C datasets
    # both_A, both_B = concat_the_AB_traces_for_symmetry(df_A=Sister.first_cousin_A_df, df_B=Sister.first_cousin_B_df)
    # Sister.plot_relationship_correlations(df_1=both_A, df_2=both_B, df_1_variables=Sister._variable_names,
    #                                       df_2_variables=Sister._variable_names,
    #                                       x_labels=Sister._variable_symbols.loc['_B, normalized lengths'],
    #                                       y_labels=Sister._variable_symbols.loc['_A, normalized lengths'])
    #
    # both_A, both_B = concat_the_AB_traces_for_symmetry(df_A=Nonsister.first_cousin_A_df, df_B=Nonsister.first_cousin_B_df)
    # Nonsister.plot_relationship_correlations(df_1=both_A, df_2=both_B, df_1_variables=Nonsister._variable_names,
    #                                          df_2_variables=Nonsister._variable_names,
    #                                          x_labels=Nonsister._variable_symbols.loc['_B, normalized lengths'],
    #                                          y_labels=Nonsister._variable_symbols.loc['_A, normalized lengths'])
    #
    # both_A, both_B = concat_the_AB_traces_for_symmetry(df_A=Control.first_cousin_A_df, df_B=Control.first_cousin_B_df)
    # Control.plot_relationship_correlations(df_1=both_A, df_2=both_B, df_1_variables=Control._variable_names,
    #                                        df_2_variables=Control._variable_names,
    #                                        x_labels=Control._variable_symbols.loc['_B, normalized lengths'],
    #                                        y_labels=Control._variable_symbols.loc['_A, normalized lengths'])


if __name__ == '__main__':
    # with_different_averages()
    main()
