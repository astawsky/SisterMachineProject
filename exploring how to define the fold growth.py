import pickle
import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from sklearn.linear_model import LinearRegression
import importlib


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
            if corr[0] == 'fold_growth':
                old = (mom['generationtime'] - mom['generationtime'].mean())*(mom['growth_rate'] - mom['growth_rate'].mean()) - \
                      ((mom['generationtime'] - mom['generationtime'].mean())*(mom['growth_rate'] - mom['growth_rate'].mean())).mean()
            else:
                old = mom[corr[0]] - mom[corr[0]].mean()
            if corr[1] == 'fold_growth':
                new = (daug['generationtime'] - daug['generationtime'].mean())*(daug['growth_rate'] - daug['growth_rate'].mean()) - \
                      ((daug['generationtime'] - daug['generationtime'].mean())*(daug['growth_rate'] - daug['growth_rate'].mean())).mean()
            else:
                new = daug[corr[1]] - daug[corr[1]].mean()
            
            r, low, high = pearson_corr_with_confidence_intervals(old, new)
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'average'],
                                                  [r, low, high, label, separation, 'global'])), ignore_index=True)

            # for the global average
            if corr[0] == 'fold_growth':
                old = (mom['generationtime'] - mom['trap_avg_generationtime'].mean()) * (mom['growth_rate'] - mom['trap_avg_growth_rate'].mean()) - \
                      ((mom['generationtime'] - mom['trap_avg_generationtime'].mean()) * (mom['growth_rate'] - mom['trap_avg_growth_rate'].mean())).mean()
            else:
                old = mom[corr[0]] - mom['trap_avg_' + corr[0]]
            if corr[1] == 'fold_growth':
                new = (daug['generationtime'] - daug['trap_avg_generationtime'].mean()) * (daug['growth_rate'] - daug['trap_avg_growth_rate'].mean()) - \
                      ((daug['generationtime'] - daug['trap_avg_generationtime'].mean()) * (daug['growth_rate'] - daug['trap_avg_growth_rate'].mean())).mean()
            else:
                new = daug[corr[1]] - daug['trap_avg_' + corr[1]]
            
            r, low, high = pearson_corr_with_confidence_intervals(old, new)
            all_corrs = all_corrs.append(dict(zip(['correlation coefficient', 'low', 'high', 'corr', 'relation', 'average'],
                                                  [r, low, high, label, separation, 'trap'])), ignore_index=True)

            # for the global average
            if corr[0] == 'fold_growth':
                old = (mom['generationtime'] - mom['traj_avg_generationtime'].mean()) * (mom['growth_rate'] - mom['traj_avg_growth_rate'].mean()) - \
                      ((mom['generationtime'] - mom['traj_avg_generationtime'].mean()) * (
                                  mom['growth_rate'] - mom['traj_avg_growth_rate'].mean())).mean()
            else:
                old = mom[corr[0]] - mom['traj_avg_' + corr[0]]
            if corr[1] == 'fold_growth':
                new = (daug['generationtime'] - daug['traj_avg_generationtime'].mean()) * (
                            daug['growth_rate'] - daug['traj_avg_growth_rate'].mean()) - \
                      ((daug['generationtime'] - daug['traj_avg_generationtime'].mean()) * (
                                  daug['growth_rate'] - daug['traj_avg_growth_rate'].mean())).mean()
            else:
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
    plt.show()
    # plt.savefig('New Phi, Inter-Generational Correlations, all averages', dpi=300)





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

    mom = Population.mother_dfs[0].copy()
    for col in Population._variable_names:
        mom[col] = mom[col] - mom['trap_avg_' + col]
        # if col != 'fold_growth':
        #     mom[col] = mom[col] - mom['trap_avg_'+col]
        #     print(col)


    # mom = mom[Population._variable_names]

    new = pd.DataFrame(columns=[r'$\delta \tau \delta \alpha$', r'$\phi$'])
    new[r'$\delta \tau \delta \alpha$'] = mom['generationtime']*mom['growth_rate']
    new[r'$\phi$'] = mom['fold_growth']
    new[r'$\bar{\tau}\bar{\alpha}$'] = mom['trap_avg_generationtime'] * mom['trap_avg_growth_rate']
    new[r'$\bar{phi}$'] = mom['trap_avg_fold_growth']
    new[r'$\bar{\alpha}\delta\tau$'] = mom['trap_avg_growth_rate'] * mom['generationtime']
    new[r'$\bar{\tau}\delta\alpha$'] = mom['trap_avg_generationtime'] * mom['growth_rate']

    new['something'] = mom['generationtime']*mom['growth_rate']+mom['trap_avg_growth_rate']*mom['generationtime']+mom['trap_avg_generationtime']*mom['growth_rate']

    # mother_dfs, daughter_dfs = Population.intergenerational_dataframe_creations(all_data_dict=Population._log_all_data_dict, how_many_separations=6)
    #
    # mom = mother_dfs[0]
    # daug = daughter_dfs[0]
    #
    # trap_1 = daug['length_birth'] - mom['length_birth']
    # trap_2 = mom['division_ratio']+mom['trap_avg_division_ratio']+(mom['growth_rate']+mom['trap_avg_growth_rate'])*()

    # sns.regplot(data=mom, x='generationtime', y='growth_rate', label=str(round(stats.pearsonr(mom['generationtime'], mom['growth_rate'])[0], 3)))
    # plt.legend()
    # plt.show()
    #
    # sns.regplot(data=new, x=r'$\delta \tau \delta \alpha$', y=r'$\phi$', label=str(round(stats.pearsonr(new[r'$\delta \tau \delta \alpha$'], new[r'$\phi$'])[0], 3)))
    # plt.legend()
    # plt.show()

    # """ this one is really high! """
    # sns.regplot(data=new, x=r'$\bar{\tau}\bar{\alpha}$', y=r'$\bar{phi}$',
    #             label=str(round(stats.pearsonr(new[r'$\bar{\tau}\bar{\alpha}$'], new[r'$\bar{phi}$'])[0], 3)))
    # plt.legend()
    # plt.show()
    #
    # sns.regplot(data=new, x=r'$\bar{\alpha}\delta\tau$', y=r'$\bar{\tau}\delta\alpha$',
    #             label=str(round(stats.pearsonr(new[r'$\bar{\alpha}\delta\tau$'], new[r'$\bar{\tau}\delta\alpha$'])[0], 3)))
    # plt.legend()
    # plt.show()

    sns.regplot(data=new, x='something', y=r'$\phi$',
                label=str(round(stats.pearsonr(new['something'], new[r'$\phi$'])[0], 3)))
    plt.legend()
    plt.show()


    # inter_generational_correlations_all_avgs(Population)

    # importlib.import_module('Graph of relationship pearson coefficients and confidence intervals.py')






if __name__ == '__main__':
    main()
