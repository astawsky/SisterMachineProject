import pickle
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.linear_model import LinearRegression


def concat_the_AB_traces_for_symmetry(df_A, df_B):
    both_A = pd.concat([df_A.copy(), df_B.copy()], axis=0).reset_index(drop=True)
    both_B = pd.concat([df_B.copy(), df_A.copy()], axis=0).reset_index(drop=True)
    return both_A, both_B


def pearson_corr_with_confidence_intervals(vec1, vec2):
    # Get the Pearson Correlation
    r = stats.pearsonr(vec1, vec2)[0]
    # Get the confidence interval:
    # Use the Fisher transformation to get z
    z = np.arctanh(r)
    # sigma value is the standard error
    sigma = (1/((len(vec1)-3)**0.5))
    # get a 95% confidence interval
    cint = z + np.array([-1, 1]) * sigma * stats.norm.ppf((1 + 0.95) / 2)
    # get the interval
    low = np.tanh(cint)[0]
    high = np.tanh(cint)[1]
    return r, low, high


def plot_intra(correlation_coefficient, low, high, corr, relation, dataset, **kwargs):
    data = pd.DataFrame(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                 [correlation_coefficient, low, high, corr, relation, dataset])))

    ax = plt.gca()

    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)

    if data['dataset'].iloc[0] == 'Sister':
        x_coords = x_coords[:len(data)]
        y_coords = y_coords[:len(data)]
    if data['dataset'].iloc[0] == 'Nonsister':
        x_coords = x_coords[len(data):2*len(data)]
        y_coords = y_coords[len(data):2*len(data)]
    if data['dataset'].iloc[0] == 'Control':
        x_coords = x_coords[2*len(data):3*len(data)]
        y_coords = y_coords[2*len(data):3*len(data)]

    # # get the fit
    # reg = LinearRegression(fit_intercept=False).fit(np.array(x_coords).reshape(-1, 1), np.log(np.abs(np.array(y_coords))))
    # ax.plot(x_coords, reg.predict([[0]]) + np.exp(reg.coef_ * x_coords), ls='--')

    # print(reg.intercept_, reg.coef_)
    # print(y_coords, reg.predict(np.array(x_coords).reshape(-1, 1)))

    # Calculate the type of error to plot as the error bars
    # Make sure the order is the same as the points were looped over
    low_row = np.array(data['low'])
    high_row = np.array(data['high'])
    ax.axhline(y=0, color='black', ls='--')
    ax.errorbar(x_coords, y_coords, yerr=[low_row, high_row], capsize=1.7, barsabove=True)


def intra_generational_correlations(Sister, Nonsister, Control):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Sister._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Sister._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_A, ${}$_B)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'])

    for separation in range(len(Sister.A_intra_gen_bacteria)):

        sis_A, sis_B = concat_the_AB_traces_for_symmetry(Sister.A_intra_gen_bacteria[separation], Sister.B_intra_gen_bacteria[separation])
        non_sis_A, non_sis_B = concat_the_AB_traces_for_symmetry(Nonsister.A_intra_gen_bacteria[separation], Nonsister.B_intra_gen_bacteria[separation])
        control_A, control_B = concat_the_AB_traces_for_symmetry(Control.A_intra_gen_bacteria[separation], Control.B_intra_gen_bacteria[separation])

        for corr, label in zip(corrs, y_labels):
            r, low, high = pearson_corr_with_confidence_intervals(sis_A[corr[0]], sis_B[corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low-r), np.abs(high-r), label, separation, 'Sister'])), ignore_index=True)
            r, low, high = pearson_corr_with_confidence_intervals(non_sis_A[corr[0]], non_sis_B[corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low-r), np.abs(high-r), label, separation, 'Nonsister'])), ignore_index=True)
            r, low, high = pearson_corr_with_confidence_intervals(control_A[corr[0]], control_B[corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low-r), np.abs(high-r), label, separation, 'Control'])), ignore_index=True)

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6, hue='dataset')
    g.map(sns.pointplot, 'relation', 'correlation coefficient', dodge=True, join=True, ci=None)
    g.map(plot_intra, 'correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset')
    g.add_legend()
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Sister._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel+r'$_A$')
    for ind, xlabel in zip(range(30, 37), Sister._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel+r'$_B$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='x-large')
        # g.axes[col].set_xlabel('')
    h = plt.gca().get_lines()
    g.add_legend(legend_data=dict(zip(['Sister', 'Nonsister', 'Control'], h[:3])))
    # plt.show()
    plt.savefig('Intra-Generational Correlations', dpi=300)


def intra_generational_correlations_global(Sister, Nonsister, Control, global_means):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Sister._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Sister._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_A, ${}$_B)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'])

    for separation in range(len(Sister.A_intra_gen_bacteria)):

        sis_A, sis_B = concat_the_AB_traces_for_symmetry(Sister.A_intra_gen_bacteria[separation], Sister.B_intra_gen_bacteria[separation])
        non_sis_A, non_sis_B = concat_the_AB_traces_for_symmetry(Nonsister.A_intra_gen_bacteria[separation], Nonsister.B_intra_gen_bacteria[separation])
        control_A, control_B = concat_the_AB_traces_for_symmetry(Control.A_intra_gen_bacteria[separation], Control.B_intra_gen_bacteria[separation])

        for corr, label in zip(corrs, y_labels):
            r, low, high = pearson_corr_with_confidence_intervals(sis_A[corr[0]]-global_means[corr[0]], sis_B[corr[1]]-global_means[corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low-r), np.abs(high-r), label, separation, 'Sister'])), ignore_index=True)
            r, low, high = pearson_corr_with_confidence_intervals(non_sis_A[corr[0]]-global_means[corr[0]], non_sis_B[corr[1]]-global_means[corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low-r), np.abs(high-r), label, separation, 'Nonsister'])), ignore_index=True)
            r, low, high = pearson_corr_with_confidence_intervals(control_A[corr[0]]-global_means[corr[0]], control_B[corr[1]]-global_means[corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low-r), np.abs(high-r), label, separation, 'Control'])), ignore_index=True)

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6, hue='dataset')
    g.map(sns.pointplot, 'relation', 'correlation coefficient', dodge=True, join=True, ci=None)
    g.map(plot_intra, 'correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset')
    g.add_legend()
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Sister._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel+r'$_A$')
    for ind, xlabel in zip(range(30, 37), Sister._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel+r'$_B$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='x-large')
        # g.axes[col].set_xlabel('')
    h = plt.gca().get_lines()
    g.add_legend(legend_data=dict(zip(['Sister', 'Nonsister', 'Control'], h[:3])))
    # plt.show()
    plt.savefig('Intra-Generational Correlations, global', dpi=300)


def intra_generational_correlations_trap(Sister, Nonsister, Control):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Sister._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Sister._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_A, ${}$_B)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'])

    for separation in range(len(Sister.A_intra_gen_bacteria)):

        sis_A, sis_B = concat_the_AB_traces_for_symmetry(Sister.A_intra_gen_bacteria[separation], Sister.B_intra_gen_bacteria[separation])
        non_sis_A, non_sis_B = concat_the_AB_traces_for_symmetry(Nonsister.A_intra_gen_bacteria[separation], Nonsister.B_intra_gen_bacteria[separation])
        control_A, control_B = concat_the_AB_traces_for_symmetry(Control.A_intra_gen_bacteria[separation], Control.B_intra_gen_bacteria[separation])

        for corr, label in zip(corrs, y_labels):
            r, low, high = pearson_corr_with_confidence_intervals(sis_A[corr[0]]-sis_A['trap_avg_'+corr[0]], sis_B[corr[1]]-sis_B['trap_avg_'+corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low-r), np.abs(high-r), label, separation, 'Sister'])), ignore_index=True)
            r, low, high = pearson_corr_with_confidence_intervals(non_sis_A[corr[0]]-non_sis_A['trap_avg_'+corr[0]], non_sis_B[corr[1]]-non_sis_B['trap_avg_'+corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low-r), np.abs(high-r), label, separation, 'Nonsister'])), ignore_index=True)
            r, low, high = pearson_corr_with_confidence_intervals(control_A[corr[0]]-control_A['trap_avg_'+corr[0]], control_B[corr[1]]-control_B['trap_avg_'+corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low-r), np.abs(high-r), label, separation, 'Control'])), ignore_index=True)

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6, hue='dataset')
    g.map(sns.pointplot, 'relation', 'correlation coefficient', dodge=True, join=True, ci=None)
    g.map(plot_intra, 'correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset')
    g.add_legend()
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Sister._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel+r'$_A$')
    for ind, xlabel in zip(range(30, 37), Sister._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel+r'$_B$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='x-large')
        # g.axes[col].set_xlabel('')
    h = plt.gca().get_lines()
    g.add_legend(legend_data=dict(zip(['Sister', 'Nonsister', 'Control'], h[:3])))
    # plt.show()
    plt.savefig('Intra-Generational Correlations, trap', dpi=300)


def intra_generational_correlations_traj(Sister, Nonsister, Control):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Sister._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Sister._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_A, ${}$_B)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'])

    for separation in range(len(Sister.A_intra_gen_bacteria)):

        print(set(list(Control.reference_A_dict.values())))
        print(len(set(list(Control.reference_A_dict.values()))))
        print(set(list(Control.reference_B_dict.values())))
        print(len(set(list(Control.reference_B_dict.values()))))
        exit()

        sis_A, sis_B = concat_the_AB_traces_for_symmetry(Sister.A_intra_gen_bacteria[separation], Sister.B_intra_gen_bacteria[separation])
        non_sis_A, non_sis_B = concat_the_AB_traces_for_symmetry(Nonsister.A_intra_gen_bacteria[separation], Nonsister.B_intra_gen_bacteria[separation])
        control_A, control_B = concat_the_AB_traces_for_symmetry(Control.A_intra_gen_bacteria[separation], Control.B_intra_gen_bacteria[separation])

        for corr, label in zip(corrs, y_labels):
            r, low, high = pearson_corr_with_confidence_intervals(sis_A[corr[0]] - sis_A['traj_avg_' + corr[0]],
                                                                  sis_B[corr[1]] - sis_B['traj_avg_' + corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low - r), np.abs(high - r), label, separation, 'Sister'])), ignore_index=True)
            r, low, high = pearson_corr_with_confidence_intervals(non_sis_A[corr[0]] - non_sis_A['traj_avg_' + corr[0]],
                                                                  non_sis_B[corr[1]] - non_sis_B['traj_avg_' + corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low - r), np.abs(high - r), label, separation, 'Nonsister'])), ignore_index=True)
            r, low, high = pearson_corr_with_confidence_intervals(control_A[corr[0]] - control_A['traj_avg_' + corr[0]],
                                                                  control_B[corr[1]] - control_B['traj_avg_' + corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
                                                  [r, np.abs(low - r), np.abs(high - r), label, separation, 'Control'])), ignore_index=True)

            # r, low, high = pearson_corr_with_confidence_intervals(sis_A[corr[0]]-Sister.A_intra_gen_bacteria[separation][corr[0]].mean(), sis_B[corr[1]]-Sister.B_intra_gen_bacteria[separation][corr[1]].mean())
            # all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
            #                                       [r, np.abs(low-r), np.abs(high-r), label, separation, 'Sister'])), ignore_index=True)
            # r, low, high = pearson_corr_with_confidence_intervals(non_sis_A[corr[0]]-Nonsister.A_intra_gen_bacteria[separation][corr[0]].mean(), non_sis_B[corr[1]]-Nonsister.B_intra_gen_bacteria[separation][corr[1]].mean())
            # all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
            #                                       [r, np.abs(low-r), np.abs(high-r), label, separation, 'Nonsister'])), ignore_index=True)
            # r, low, high = pearson_corr_with_confidence_intervals(control_A[corr[0]]-Control.A_intra_gen_bacteria[separation][corr[0]], control_B[corr[1]]-Control.B_intra_gen_bacteria[separation][corr[1]])
            # all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset'],
            #                                       [r, np.abs(low-r), np.abs(high-r), label, separation, 'Control'])), ignore_index=True)

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6, hue='dataset')
    g.map(sns.pointplot, 'relation', 'correlation coefficient', dodge=True, join=True, ci=None)
    g.map(plot_intra, 'correlation coefficient', 'low', 'high', 'corr', 'relation', 'dataset')
    g.add_legend()
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Sister._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel+r'$_A$')
    for ind, xlabel in zip(range(30, 37), Sister._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel+r'$_B$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='x-large')
        # g.axes[col].set_xlabel('')
    h = plt.gca().get_lines()
    g.add_legend(legend_data=dict(zip(['Sister', 'Nonsister', 'Control'], h[:3])))
    # plt.show()
    plt.savefig('Intra-Generational Correlations, traj', dpi=300)


def plot_it(coeff, corr, high, low, relations, **kwargs):
    ax = plt.gca()
    ax.axhline(y=0, color='black', ls='--')
    yerr = np.array([np.abs(np.array(low - coeff)), np.abs(np.array(high - coeff))])
    ax.errorbar(x=[str(ind) for ind in range(1, len(relations)+1)], y=coeff, yerr=yerr, marker='.', capsize=1.7, barsabove=True)
    relations = relations.reset_index(drop=True)
    # making sure we are plotting the points in the corresponding order
    for separation in range(len(relations)):
        if relations.loc[separation] != separation:
            print('Error '+str(separation)+' separation is not in '+str(separation)+' index')


def inter_generational_correlations(Population):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Population._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Population._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_m, ${}$_d)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation'])

    for separation in range(len(Population.mother_dfs)):
        for corr, label in zip(corrs, y_labels):
            r, low, high = pearson_corr_with_confidence_intervals(Population.mother_dfs[separation][corr[0]],
                                                                  Population.daughter_dfs[separation][corr[1]])
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation'], 
                                                  [r, low, high, label, separation])), ignore_index=True)

    # sanity check to make sure the confidence intervals make sense
    for index in range(len(all_corrs)):
        if all_corrs['correlation coefficient'].iloc[index] < all_corrs['low'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is lower than', all_corrs['low'].iloc[index])
            print(all_corrs.iloc[index])
        if all_corrs['correlation coefficient'].iloc[index] > all_corrs['high'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is greater than', all_corrs['high'].iloc[index])
            print(all_corrs.iloc[index])

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6)
    g.map(plot_it, 'correlation coefficient', 'corr', 'high', 'low', 'relation')
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Population._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel + r'$_m$')
    for ind, xlabel in zip(range(30, 37), Population._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel + r'$_d$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='xx-large')
        # g.axes[col].set_xlabel('')
    # save the figure we just made
    # plt.show()
    plt.savefig('Inter-Generational Correlations, new and more', dpi=300)


def plot_it_all_avgs(coeff, corr, high, low, relations, **kwargs):
    ax = plt.gca()
    ax.axhline(y=0, color='black', ls='--')
    yerr = np.array([np.abs(np.array(low - coeff)), np.abs(np.array(high - coeff))])
    ax.errorbar(x=[str(ind) for ind in range(1, len(relations)+1)], y=coeff, yerr=yerr, marker='.', capsize=1.7, barsabove=True)
    relations = relations.reset_index(drop=True)
    # making sure we are plotting the points in the corresponding order
    for separation in range(len(relations)):
        if relations.loc[separation] != separation:
            print('Error '+str(separation)+' separation is not in '+str(separation)+' index')


def inter_generational_correlations_all_avgs(Population):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Population._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Population._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_m, ${}$_d)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation', 'average'])

    for separation in range(len(Population.mother_dfs)):
        mom = Population.mother_dfs[separation].copy()
        daug = Population.daughter_dfs[separation].copy()

        mom['length_birth'] = np.log(mom['length_birth'])
        daug['length_birth'] = np.log(daug['length_birth'])
        mom['trap_avg_length_birth'] = np.log(mom['trap_avg_length_birth'])
        daug['trap_avg_length_birth'] = np.log(daug['trap_avg_length_birth'])
        mom['traj_avg_length_birth'] = np.log(mom['traj_avg_length_birth'])
        daug['traj_avg_length_birth'] = np.log(daug['traj_avg_length_birth'])
        for corr, label in zip(corrs, y_labels):
            # for the global average
            old = mom[corr[0]] - mom[corr[0]].mean()
            new = daug[corr[1]] - daug[corr[1]].mean()
            r, low, high = pearson_corr_with_confidence_intervals(old, new)
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'average'],
                                                  [r, low, high, label, separation, 'global'])), ignore_index=True)

            # for the trap average
            old = mom[corr[0]] - mom['trap_avg_' + corr[0]]
            new = daug[corr[1]] - daug['trap_avg_' + corr[1]]
            r, low, high = pearson_corr_with_confidence_intervals(old, new)
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'average'],
                                                  [r, low, high, label, separation, 'trap'])), ignore_index=True)

            # for the trajectory average
            old = mom[corr[0]] - mom['traj_avg_' + corr[0]]
            new = daug[corr[1]] - daug['traj_avg_' + corr[1]]
            r, low, high = pearson_corr_with_confidence_intervals(old, new)
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'average'],
                                                  [r, low, high, label, separation, 'traj'])), ignore_index=True)

    # sanity check to make sure the confidence intervals make sense
    for index in range(len(all_corrs)):
        if all_corrs['correlation coefficient'].iloc[index] < all_corrs['low'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is lower than', all_corrs['low'].iloc[index])
            print(all_corrs.iloc[index])
        if all_corrs['correlation coefficient'].iloc[index] > all_corrs['high'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is greater than', all_corrs['high'].iloc[index])
            print(all_corrs.iloc[index])

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6, hue="average")
    g.map(plot_it_all_avgs, 'correlation coefficient', 'corr', 'high', 'low', 'relation')
    # g.add_legend()
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Population._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel + r'$_m$')
    for ind, xlabel in zip(range(30, 37), Population._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel + r'$_d$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='xx-large')
        # g.axes[col].set_xlabel('')
    # g.legend_.remove()
    # h = [blah.get_color() for blah in plt.gca().get_lines()]
    # g.add_legend(legend_data=dict(zip(['Global', 'Trap', 'Traj'], h[:3])))

    h = sns.color_palette("deep").as_hex()[:3]

    name_to_color = dict(zip(['Global', 'Trap', 'Traj'], h))
    handles = [matplotlib.patches.Patch(color=v, label=k) for k, v in name_to_color.items()]
    print(dict(zip(['Global', 'Trap', 'Traj'], handles[:3])))
    # g.add_legend(legend_data=dict(zip(['Global', 'Trap', 'Traj'], handles[:3])))
    g = g.add_legend(legend_data=name_to_color)
    # save the figure we just made
    # plt.show()
    plt.savefig('Inter-Generational Correlations, all averages', dpi=300)


def inter_generational_correlations_trap_avgs(Population):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Population._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Population._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_m, ${}$_d)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation'])

    for separation in range(len(Population.mother_dfs)):
        mom = Population.mother_dfs[separation].copy()
        daug = Population.daughter_dfs[separation].copy()

        mom['length_birth'] = np.log(mom['length_birth'])
        daug['length_birth'] = np.log(daug['length_birth'])
        mom['trap_avg_length_birth'] = np.log(mom['trap_avg_length_birth'])
        daug['trap_avg_length_birth'] = np.log(daug['trap_avg_length_birth'])
        for corr, label in zip(corrs, y_labels):
            old = mom[corr[0]] - mom['trap_avg_'+corr[0]]
            new = daug[corr[1]] - daug['trap_avg_' + corr[1]]
            r, low, high = pearson_corr_with_confidence_intervals(old, new)
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation'],
                                                  [r, low, high, label, separation])), ignore_index=True)

    # sanity check to make sure the confidence intervals make sense
    for index in range(len(all_corrs)):
        if all_corrs['correlation coefficient'].iloc[index] < all_corrs['low'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is lower than', all_corrs['low'].iloc[index])
            print(all_corrs.iloc[index])
        if all_corrs['correlation coefficient'].iloc[index] > all_corrs['high'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is greater than', all_corrs['high'].iloc[index])
            print(all_corrs.iloc[index])

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6)
    g.map(plot_it, 'correlation coefficient', 'corr', 'high', 'low', 'relation')
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Population._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel + r'$_m$')
    for ind, xlabel in zip(range(30, 37), Population._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel + r'$_d$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='xx-large')
        # g.axes[col].set_xlabel('')
    # save the figure we just made
    # plt.show()
    plt.savefig('Inter-Generational Correlations, trap average', dpi=300)


def inter_generational_correlations_global_avgs(Population):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Population._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Population._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_m, ${}$_d)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation'])

    for separation in range(len(Population.mother_dfs)):
        mom = Population.mother_dfs[separation].copy()
        daug = Population.daughter_dfs[separation].copy()

        mom['length_birth'] = np.log(mom['length_birth'])
        daug['length_birth'] = np.log(daug['length_birth'])
        for corr, label in zip(corrs, y_labels):
            old = mom[corr[0]] - mom[corr[0]].mean()
            new = daug[corr[1]] - daug[corr[1]].mean()
            r, low, high = pearson_corr_with_confidence_intervals(old, new)
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation'],
                                                  [r, low, high, label, separation])), ignore_index=True)

    # sanity check to make sure the confidence intervals make sense
    for index in range(len(all_corrs)):
        if all_corrs['correlation coefficient'].iloc[index] < all_corrs['low'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is lower than', all_corrs['low'].iloc[index])
            print(all_corrs.iloc[index])
        if all_corrs['correlation coefficient'].iloc[index] > all_corrs['high'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is greater than', all_corrs['high'].iloc[index])
            print(all_corrs.iloc[index])

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6)
    g.map(plot_it, 'correlation coefficient', 'corr', 'high', 'low', 'relation')
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Population._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel + r'$_m$')
    for ind, xlabel in zip(range(30, 37), Population._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel + r'$_d$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='xx-large')
        # g.axes[col].set_xlabel('')
    # save the figure we just made
    # plt.show()
    plt.savefig('Inter-Generational Correlations, global average', dpi=300)


def inter_generational_correlations_traj_avgs(Population):
    # collect all possible permutations of two variables which is the same as all possible correlations one can do with these variables
    corrs = [p for p in itertools.product(Population._variable_names, repeat=2)]
    corrs_sym = [p for p in itertools.product(Population._variable_symbols.loc['without subscript'], repeat=2)]
    y_labels = [r'$\rho(${}$_m, ${}$_d)$'.format(corr[0], corr[1]) for corr in corrs_sym]

    all_corrs = pd.DataFrame(columns=['correlation coefficient', 'low', 'high', 'corr', 'relation'])

    for separation in range(len(Population.mother_dfs)):
        mom = Population.mother_dfs[separation].copy()
        daug = Population.daughter_dfs[separation].copy()

        mom['length_birth'] = np.log(mom['length_birth'])
        daug['length_birth'] = np.log(daug['length_birth'])
        mom['traj_avg_length_birth'] = np.log(mom['traj_avg_length_birth'])
        daug['traj_avg_length_birth'] = np.log(daug['traj_avg_length_birth'])
        for corr, label in zip(corrs, y_labels):
            old = mom[corr[0]] - mom['traj_avg_'+corr[0]]
            new = daug[corr[1]] - daug['traj_avg_' + corr[1]]
            r, low, high = pearson_corr_with_confidence_intervals(old, new)
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation'],
                                                  [r, low, high, label, separation])), ignore_index=True)

    # sanity check to make sure the confidence intervals make sense
    for index in range(len(all_corrs)):
        if all_corrs['correlation coefficient'].iloc[index] < all_corrs['low'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is lower than', all_corrs['low'].iloc[index])
            print(all_corrs.iloc[index])
        if all_corrs['correlation coefficient'].iloc[index] > all_corrs['high'].iloc[index]:
            print('index', index)
            print(all_corrs['correlation coefficient'].iloc[index], 'is greater than', all_corrs['high'].iloc[index])
            print(all_corrs.iloc[index])

    # set the background style of the plots
    sns.set_style('whitegrid')
    # plot the correlations
    g = sns.FacetGrid(all_corrs, col="corr", col_wrap=6)
    g.map(plot_it, 'correlation coefficient', 'corr', 'high', 'low', 'relation')
    # make sure we have comprehensive y labels where we need to and no x_labels anywhere except for the tick marks
    for row, ylabel in zip(range(0, 37, 6), Population._variable_symbols.loc['without subscript']):
        g.axes[row].set_ylabel(ylabel + r'$_m$')
    for ind, xlabel in zip(range(30, 37), Population._variable_symbols.loc['without subscript']):
        g.axes[ind].set_xlabel(xlabel + r'$_d$')
    for col, label in zip(range(36), y_labels):
        g.axes[col].set_title('')
        g.axes[col].annotate(label, xy=(.1, .9), xycoords=g.axes[col].transAxes, weight='extra bold', size='xx-large')
        # g.axes[col].set_xlabel('')
    # save the figure we just made
    # plt.show()
    plt.savefig('Inter-Generational Correlations, traj average', dpi=300)


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

    print(Population.mother_dfs[0].columns)

    global_means = Population.mother_dfs[0][Population._variable_names].mean()

    # intra_generational_correlations_global(Sister, Nonsister, Control, global_means)
    # intra_generational_correlations_trap(Sister, Nonsister, Control)
    intra_generational_correlations_traj(Sister, Nonsister, Control)

    # inter_generational_correlations_all_avgs(Population)

    # intra_generational_correlations(Sister, Nonsister, Control)
    # inter_generational_correlations_global_avgs(Population)
    # inter_generational_correlations_trap_avgs(Population)
    # inter_generational_correlations_traj_avgs(Population)
    # # inter_generational_correlations(Population)


if __name__ == '__main__':
    main()
