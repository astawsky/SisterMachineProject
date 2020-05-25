
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys, math
import glob
import pickle
import os
import scipy.stats as stats
import random
from sklearn.linear_model import LinearRegression
import seaborn as sns
import NewSisterCellClass as ssc


""" For making the Seaborn Plots look better """


def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(r"$\rho = {:.2f}$".format(r),
                xy=(.1, .9), xycoords=ax.transAxes, fontweight='bold', fontsize=16)


""" For making the Seaborn Plots look better """


def limit_the_axes(x, y, **kws):
    ax = plt.gca()
    ax.set_xlim([-3 * np.std(x) + np.mean(x), 5 * np.std(x) + np.mean(x)])
    ax.set_ylim([-3 * np.std(y) + np.mean(y), 5 * np.std(y) + np.mean(y)])


def plot_the_label(var1, **kwargs):
    ax = plt.gca()
    ax.annotate(r'${:.3}\pm{:.3}$'.format(np.mean(var1), np.std(var1)),
                xy=(.1, .9), xycoords=ax.transAxes, fontweight='bold')


""" For seeing the whole dataframe in the console """


def print_full_dataframe(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


class SisterCellData(object):
    _variable_names = ['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio', 'added_length']
    _trap_variable_names = ['trap_avg_' + name for name in _variable_names]
    _log_trap_variable_names = ['log_trap_avg_' + name for name in _variable_names]
    _traj_variable_names = ['traj_avg_' + name for name in _variable_names]
    _log_traj_variable_names = ['log_traj_avg_' + name for name in _variable_names]
    _relationship_dfs_variable_names = _variable_names + _trap_variable_names + _traj_variable_names + _log_trap_variable_names + _log_traj_variable_names

    # _variable_symbols = pd.DataFrame(columns=_variable_names, index=['without subscript', '_n', '_{n+1}', 'without subscript, normalized lengths',
    #                     '_n, normalized lengths', '_{n+1}, normalized lengths', 'without subscript, normalized lengths, delta phi',
    #                     '_n normalized lengths, delta phi', '_{n+1} normalized lengths, delta phi'])
    #
    # # same as above but formatted in latex to look better in plots
    # _variable_symbols.loc['_n'] = [r'$\tau_n$', r'$x_n(0)$', r'$x_n(\tau)$', r'$\alpha_n$', r'$\phi_n$', r'$f_n$']
    # _variable_symbols.loc['_{n+1}'] = [r'$\tau_{n+1}$', r'$x_{n+1}(0)$', r'$x_{n+1}(\tau)$', r'$\alpha_{n+1}$', r'$\phi_{n+1}$', r'$f_{n+1}$']
    # _variable_symbols.loc['without subscript'] = [r'$\tau$', r'$\ln(\frac{x(0)}{x^*})$', r'$\ln(\frac{x(\tau)}{x^*})$',
    #                                               r'$\alpha$', r'$\phi$', r'$f$']
    #
    # _variable_symbols.loc['_A, normalized lengths'] = [r'$\tau_A$', r'$\ln(\frac{x_A(0)}{x^*})$', r'$\ln(\frac{x_A(\tau)}{x^*})$',
    #                                                    r'$\alpha_A$', r'$\phi_A$', r'$f_A$']
    # _variable_symbols.loc['_B, normalized lengths'] = [r'$\tau_B$', r'$\ln(\frac{x_B(0)}{x^*})$', r'$\ln(\frac{x_B(\tau)}{x^*})$',
    #                                                    r'$\alpha_B$', r'$\phi_B$', r'$f_B$']
    #
    # _variable_symbols.loc['_n, normalized lengths'] = [r'$\tau_n$', r'$\ln(\frac{x_n(0)}{x^*})$', r'$\ln(\frac{x_n(\tau)}{x^*})$',
    #                                                    r'$\alpha_n$', r'$\phi_n$', r'$f_n$']
    # _variable_symbols.loc['_{n+1}, normalized lengths'] = [r'$\tau_{n+1}$', r'$\ln(\frac{x_{n+1}(0)}{x^*})$', r'$\ln(\frac{x_{n+1}(\tau)}{x^*})$',
    #                                    r'$\alpha_{n+1}$', r'$\phi_{n+1}$', r'$f_{n+1}$']
    # _variable_symbols.loc['without subscript, normalized lengths'] = [r'$\tau$', r'$x(0)$', r'$x(\tau)$', r'$\alpha$', r'$\phi$', r'$f$']
    #
    # _variable_symbols.loc['_n, normalized lengths, delta phi'] = [r'\tau_n', r'\ln(\frac{x_n(0)}{x^*})', r'\ln(\frac{x_n(\tau)}{x^*})',
    #                                                               r'\alpha_n', r'\delta\phi_n', r'f_n']
    # _variable_symbols.loc['_{n+1}, normalized lengths, delta phi'] = [r'\tau_{n+1}', r'\ln(\frac{x_{n+1}(0)}{x^*})',
    #                                 r'\ln(\frac{x_{n+1}(\tau)}{x^*})', r'\alpha_{n+1}', r'\delta\phi_{n+1}', r'f_{n+1}']
    # _variable_symbols.loc['without subscript, normalized lengths, delta phi'] = [r'\tau', r'x(0)', r'x(\tau)', r'\alpha', r'\delta\phi', r'f']

    """ This is made to have as a table to look at for reference as to what functions we can use on an instance and what attributes we can use """
    def table_with_attributes_and_available_methods(self):
        method_list = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]
        dictionary = {'Class Name': self.__class__.__name__,'Attributes': list(self.__dict__.keys()), 'Methods': method_list}
        return dictionary

    """ outputs the regression matrix with the coefficients, the scores, and the intercepts of all regression of target variables from the target 
        dataframe using the factor variables from the factor dataframe """
    def linear_regression_framework(self, df_of_avgs, factor_variables, target_variables, factor_df, target_df, fit_intercept):

        # The centered variables, we are trying to find how much they depend on what centers them
        centered_fv = ['centered_'+fv for fv in factor_variables]
        centered_tv = ['centered_'+tv for tv in target_variables]

        # all the data we need to make the prediction
        factors = factor_df[factor_variables].rename(columns=dict(zip(factor_variables, centered_fv))) - \
                  df_of_avgs.rename(columns=dict(zip(list(df_of_avgs.columns), centered_fv)))

        # Make regression matrix in a dataframe representation
        df_matrix = pd.DataFrame(index=centered_tv, columns=centered_fv)

        # arrays where we will keep the information
        scores = dict()
        intercepts = dict()

        # loop over all the regressions we'll be doing
        for t_v, cntr_tv in zip(target_variables, centered_tv):
            # get the fit
            reg = LinearRegression(fit_intercept=fit_intercept).fit(factors, target_df[t_v])
            # save the information
            scores.update({t_v: reg.score(factors, target_df[t_v])})
            intercepts.update({t_v: reg.intercept_})
            # put it in the matrix
            df_matrix.loc[cntr_tv] = reg.coef_

        return df_matrix, scores, intercepts

    """ plotting the histograms of the trace and trap coefficients of variations, this is to see if in a trap/trace the generationtime varies way 
    more than the growth_rate, meaning that inside a trap the growth rate is basically constant """
    # def plot_hist_of_coefs_of_variance(self):
    #     trace_coef_var_df = pd.DataFrame(columns=['Pooling', 'Variable', 'Coefficient of Variation'])
    #     trap_coef_var_df = pd.DataFrame(columns=['Pooling', 'Variable', 'Coefficient of Variation'])
    #
    #     for var in self._variable_names:
    #         for valA, valB, valT in zip(self.A_coeffs_of_vars_dict.values(), self.B_coeffs_of_vars_dict.values(),
    #                                     self.Trap_coeffs_of_vars_dict.values()):
    #             trace_coef_var_df = trace_coef_var_df.append({'Variable': var, 'Pooling': 'trace', 'Coefficient of Variation': valA[var]},
    #                                                          ignore_index=True)
    #             trace_coef_var_df = trace_coef_var_df.append({'Variable': var, 'Pooling': 'trace', 'Coefficient of Variation': valB[var]},
    #                                                          ignore_index=True)
    #             trap_coef_var_df = trap_coef_var_df.append({'Variable': var, 'Pooling': 'trap', 'Coefficient of Variation': valT[var]},
    #                                                        ignore_index=True)
    #     coef_var_df = pd.concat([trace_coef_var_df, trap_coef_var_df], axis=0)
    #
    #     g = sns.FacetGrid(data=coef_var_df, row='Pooling', col='Variable')
    #     g = g.map(sns.distplot, 'Coefficient of Variation', kde=True)
    #     g = g.map(plot_the_label, 'Coefficient of Variation')
    #     plt.show()

    """ Creates two trajectory specific and one trap specific dictionaries (3 dictionaries in total) that contain dataframes of the mean and std 
    div of their respective data. Meant for subclasses that have the A/B distinction, ie. ONLY Sister and Nonsister subclasses """
    def trap_and_traj_variable_statistics(self, A_dict, B_dict):

        # Just a check
        if list(A_dict.keys()) != list(B_dict.keys()):
            IOError('The dictionary keys of A and B dataframes do not match! (in trap_variable_statistics method)')

        # The dictionaries that will have the trajectory and trap statistics corresponding to the keys in self.A_dict and self.B_dict
        A_stats_dict = dict()
        B_stats_dict = dict()
        trap_stats_dict = dict()

        # loop over all traps
        for key in A_dict.keys():
            # create the empty dataframe to fill
            A_stats = pd.DataFrame(columns=self._variable_names, index=['mean', 'std'])
            B_stats = pd.DataFrame(columns=self._variable_names, index=['mean', 'std'])
            trap_stats = pd.DataFrame(columns=self._variable_names, index=['mean', 'std'])

            # fill the df and save it in the dictionary
            A_stats.loc['mean'] = A_dict[key].mean()
            A_stats.loc['std'] = A_dict[key].std()
            A_stats_dict.update({key: A_stats})
            B_stats.loc['mean'] = B_dict[key].mean()
            B_stats.loc['std'] = B_dict[key].std()
            B_stats_dict.update({key: A_stats})

            # Trap specific (combining A and B)
            combined = A_dict[key].append(B_dict[key])
            trap_stats.loc['mean'] = combined.mean()
            trap_stats.loc['std'] = combined.std()
            trap_stats_dict.update({key: trap_stats})


        return A_stats_dict, B_stats_dict, trap_stats_dict

    """ Use this to get the starting and ending indices in the raw data """
    def get_division_indices(self, _data, data_id, discretize_by, ks):
        # From the raw data, we see where the difference between two values of 'discretize_by' falls drastically,
        # suggesting a division has occurred.
        diffdata = np.diff(_data[data_id][discretize_by + ks])

        # How much of a difference there has to be between two points to consider division having taken place.
        threshold_of_difference_for_division = -1

        # The array of indices where division took place.
        index_div = np.where(diffdata < threshold_of_difference_for_division)[0].flatten()

        # If the two consecutive indices in the array of indices where division took place are less than two time steps away, then we discard
        # them
        for ind1, ind2 in zip(index_div[:-1], index_div[1:]):
            if ind2 - ind1 <= 2:
                index_div = np.delete(index_div, np.where(index_div == ind2))

        # WE DISCARD THE FIRST AND LAST CYCLE BECAUSE THEY ARE MOST LIKELY NOT COMPLETE
        # THESE ARE INDICES AS WELL!

        # An array of indices where a cycle starts and ends
        start_indices = [x + 1 for x in index_div]
        end_indices = [x for x in index_div]
        start_indices.append(0)  # to count the first cycle
        start_indices.sort()  # because '0' above was added at the end and the rest are in order already
        del start_indices[-1]  # we do not count the last cycle because it most likely did not finish by the time recording ended
        end_indices.sort()  # same as with start_indices, just in case

        # If the last index is in the array of indices where a cycle starts, then it obviously a mistake and must be removed
        if _data[data_id][discretize_by + ks].index[-1] in start_indices:
            start_indices.remove(_data[data_id][discretize_by + ks].index[-1])

        # Similarly, if the fist index '0' is in the array of indices where a cycle ends, then it obviously is a mistake and must be removed
        if 0 in end_indices:
            end_indices.remove(0)

        # Sanity check: If the starting index for a cycle comes after the ending index, then it is obviously a mistake
        # Not a foolproof method so, if the length of the bacteria drops after the start time of the cycle, we know it is not the real starting
        # time since a bacteria does not shrink after dividing
        for start, end in zip(start_indices, end_indices):
            if start >= end:
                IOError('start', start, 'end', end, "didn't work")

        # Make them numpy arrays now so we can subtract them
        start_indices = np.array(start_indices)
        end_indices = np.array(end_indices)

        return start_indices, end_indices

    """ check division times, ie. plot all the A and B trajectories with the division times highlighted """
    def division_times_check(self):

        # ask if we want to save or show the division check points
        answer = input("Do you want to save the plots or do you want to see them?")
        if answer == 'save':
            name = input("What is the name of the file they should be put under?")
            os.mkdir(name)
        elif answer == 'see':
            pass
        else:
            print('wrong input! you will only see them.')

        for dataID in range(self._data_len):
            for ks in ['A', 'B']:
                # plot the trajectory and the beginning and end points
                plt.plot(self._data[dataID]['time' + ks], self._data[dataID][self._discretization_variable + ks], marker='+')
                start_indices, end_indices = self.get_division_indices(data_id=dataID, discretize_by=self._discretization_variable, ks=ks)
                plt.scatter(self._data[dataID]['time' + ks].iloc[start_indices],
                            self._data[dataID][self._discretization_variable + ks].iloc[start_indices], color='red', label='first point in cycle')
                plt.scatter(self._data[dataID]['time' + ks].iloc[end_indices],
                            self._data[dataID][self._discretization_variable + ks].iloc[end_indices], color='green', label='last point in cycle')
                plt.legend()

                # save them or show them based on what we answered
                if answer == 'save':
                    plt.savefig(name + '/' + ks + '_' + str(dataID) + '.png', dpi=300)
                elif answer == 'see':
                    plt.show()
                else:
                    plt.show()
                plt.close()


    """ Since the mean_fluorescence came later and in a different format, the dataframe creation was quick and easy since it is raw, this just splits 
    the A and B trajectories and puts them in  dictionaries, along with a dictionary containing both """
    def create_dictionaries_of_mean_fluor(self, protein_data_len, protein_data):
        """ The columns are called T, A and B to represent the time, meanfluorescence of A and B respectively """
        # creation of the dictionaries
        Both_dict = dict()
        A_dict = dict()
        B_dict = dict()

        # loop over the amount of data in the subclass
        for experiment_number in range(protein_data_len):
            TrajA = protein_data[experiment_number][['T', 'A']]
            TrajB = protein_data[experiment_number][['T', 'B']]
            TrajA = TrajA.rename(columns={'A': 'normed_fluor'})
            TrajB = TrajB.rename(columns={'B': 'normed_fluor'})
            # the general dataframe keys
            keyA = 'A_' + str(experiment_number)
            keyB = 'B_' + str(experiment_number)

            # saving the dataframes to the dictionary
            Both_dict.update({keyA: TrajA, keyB: TrajB})
            A_dict.update({str(experiment_number): TrajA})
            B_dict.update({str(experiment_number): TrajB})

        return Both_dict, A_dict, B_dict

    """ Creates the A and B trajectory dictionaries, as well as general dictionary with both the A and B dictionary elements inside, that contain the 
    trajectory generation dataframes as values and the integers as keys for A and B, and 'A_'+integer/'B_'+integer for the general dictionary """
    def create_dictionaries_of_traces(self, data_len, discretization_variable, _keylist, _data, what_to_subtract, start_index, end_index): # what_to_subtract can be either "trap/traj mean/median" or None

        """ Creates a list of two pandas dataframes, from a trap's two trajectories, that contain the values for the variables in each generation """

        def generation_dataframe_creation(dataID, _data, discretize_by='length', fit_the_length_at_birth=True):
            """
            Transform measured time series data into data for each generation:
            (1) Find cell division events as a large enough drop in the measured value given by 'discretize_by'
            (2) Compute various observables from these generations of cells
            (3) Returns two pandas dataframes for each of the discretized trajectories
            """

            # arbitrary but necessary specification options for the two cells inside the trap
            keysuffix = ['A', 'B']

            # Return two pandas-dataframes for the two trajectories usually contained in one file as two elements of a list.
            # As the two sisters do not need to have the same number of cell divisions,
            # a single dataframe might cause problems with variably lengthed trajectories.
            ret = list()

            for ks in keysuffix:

                # use this to get the starting and ending indices in the raw data
                start_indices, end_indices = self.get_division_indices(data_id=dataID, _data=_data, discretize_by=discretize_by, ks=ks)

                # How long each cycle is
                cycle_durations = np.array(_data[dataID]['time' + ks][end_indices]) - np.array(_data[dataID]['time' + ks][start_indices])

                # Store results in this dictionary, which can be easier transformed into pandas-dataframe
                ret_ks = dict()

                # Number of raw data points per generation/cycle
                data_points_per_cycle = np.rint(cycle_durations / .05) + np.ones_like(cycle_durations)

                # The x and y values per generation for the regression to get the growth rate
                domains_for_regression_per_gen = [np.linspace(_data[dataID]['time' + ks][start],
                                                              _data[dataID]['time' + ks][end], num=num_of_data_points)
                                                  for start, end, num_of_data_points in zip(start_indices, end_indices, data_points_per_cycle)]
                ranges_for_regression_per_gen = [np.log(_data[dataID][discretize_by + ks][start:end + 1])  # the end+1 is due to indexing
                                                 for start, end in zip(start_indices, end_indices)]

                # Arrays where the intercepts (the length at birth) and the slopes (the growth rate) will be stored
                regression_intercepts_per_gen = []
                regression_slopes_per_gen = []
                for domain, y_vals in zip(domains_for_regression_per_gen, ranges_for_regression_per_gen):  # loop over generations in the trace
                    # reshape the x and y values
                    domain = np.array(domain).reshape(-1, 1)
                    y_vals = np.array(y_vals).reshape(-1, 1)
                    # do the regression
                    reg = LinearRegression().fit(domain, y_vals)
                    # save them to their respective arrays
                    regression_slopes_per_gen.append(reg.coef_[0][0])
                    regression_intercepts_per_gen.append(np.exp(reg.predict(domain[0].reshape(-1, 1))[0][0]))

                # change them to numpy arrays
                regression_slopes_per_gen = np.array(regression_slopes_per_gen)
                regression_intercepts_per_gen = np.array(regression_intercepts_per_gen)

                # check if the growth_rate is negative, meaning it is obviously a not a generation
                checking_pos = np.array([slope > 0 for slope in regression_slopes_per_gen])
                if not checking_pos.all():
                    print("there's been a negative or zero growth_rate found!")
                    # change growth_rate
                    regression_slopes_per_gen = regression_slopes_per_gen[np.where(checking_pos)]
                    # change length at birth
                    regression_intercepts_per_gen = regression_intercepts_per_gen[np.where(checking_pos)]
                    # change generationtime
                    cycle_durations = cycle_durations[np.where(checking_pos)]
                    # change the indices which will change the rest
                    start_indices, end_indices = start_indices[np.where(checking_pos)], end_indices[np.where(checking_pos)]

                # check if the length_final <= length_birth, meaning it is obviously a not a generation
                if fit_the_length_at_birth:
                    checking_pos = np.array([length_birth < length_final for length_birth, length_final in
                                             zip(regression_intercepts_per_gen, np.array(_data[dataID][discretize_by + ks][end_indices]))])
                else:
                    checking_pos = np.array([length_birth < length_final for length_birth, length_final in
                                             zip(np.array(_data[dataID][discretize_by + ks][start_indices]),
                                                 np.array(_data[dataID][discretize_by + ks][end_indices]))])
                if not checking_pos.all():
                    print('length_final <= length_birth, and this so-called "generation" was taken out of its dataframe')
                    # change growth_rate
                    regression_slopes_per_gen = regression_slopes_per_gen[np.where(checking_pos)]
                    # change length at birth
                    regression_intercepts_per_gen = regression_intercepts_per_gen[np.where(checking_pos)]
                    # change generationtime
                    cycle_durations = cycle_durations[np.where(checking_pos)]
                    # change the indices which will change the rest
                    start_indices, end_indices = start_indices[np.where(checking_pos)], end_indices[np.where(checking_pos)]

                # Due to limitations of data, mainly that after some obvious division points the length of the bacteria drops, which means that it is
                # shrinking, which we assume is impossible. Either way we present is as an option, just in case.
                if fit_the_length_at_birth:
                    ret_ks['length_birth'] = regression_intercepts_per_gen
                else:
                    ret_ks['length_birth'] = np.array(_data[dataID][discretize_by + ks][start_indices])

                # Duration of growth before division
                ret_ks['generationtime'] = np.around(cycle_durations, decimals=2)

                # The measured final length before division, however it is worth to note that we have a very good approximation of this observable by
                # the mapping presented in Susman et al. (2018)
                ret_ks['length_final'] = np.array(_data[dataID][discretize_by + ks][end_indices])

                # The rate at which the bacteria grows in this cycle. NOTICE THAT THIS CHANGED FROM 'growth_length'
                ret_ks['growth_rate'] = regression_slopes_per_gen

                # The fold growth, ie. how much it grew. Defined by the rate at which it grew multiplied by the time it grew for
                # NOTE: We call this 'phi' before, in an earlier version of the code
                ret_ks['fold_growth'] = ret_ks['generationtime'] * ret_ks['growth_rate']

                # Calculating the division ratios, percentage of mother length that went to daughter
                div_rats = []
                for final, next_beg in zip(ret_ks['length_final'][:-1], ret_ks['length_birth'][1:]):
                    div_rats.append(next_beg / final)

                # we use the length at birth that is not in the dataframe in order to get enough division ratios
                div_rats.append(_data[dataID][discretize_by + ks][end_indices[-1] + 1] / ret_ks['length_final'][-1])
                ret_ks['division_ratio'] = div_rats

                # the added length to check the adder model
                ret_ks['added_length'] = ret_ks['length_final'] - ret_ks['length_birth']

                # # To do the check below
                # ret_ks_df = pd.DataFrame(ret_ks)

                # # checking that the length_final > length_birth
                # if pd.Series(ret_ks_df['length_final'] <= ret_ks_df['length_birth']).isna().any():
                #     print('length_final <= length_birth, and this so-called "generation" was taken out of its dataframe')
                #     ret_ks_df = ret_ks_df.drop(index=np.where(ret_ks_df['length_final'] <= ret_ks_df['length_birth'])[0]).reset_index(drop=True)

                # we have everything, now make a dataframe
                ret.append(pd.DataFrame(ret_ks))
            return ret
        
        def minusing(what_to_subtract, Traj_A, Traj_B):
            if what_to_subtract == None:
                # we want the measurement, so we do nothing else
                pass
            elif what_to_subtract.split(' ')[1] == 'mean':
                # we want to subtract some mean to each value
                if what_to_subtract.split(' ')[0] == 'trap':
                    trap = pd.concat([Traj_A, Traj_B], axis=0).mean()
                    Traj_A = Traj_A - trap
                    Traj_B = Traj_B - trap
                if what_to_subtract.split(' ')[0] == 'traj':
                    Traj_A = Traj_A - Traj_A.mean()
                    Traj_B = Traj_B - Traj_B.mean()
            elif what_to_subtract.split(' ')[1] == 'median':
                # we want to subtract some median to each value
                if what_to_subtract.split(' ')[0] == 'trap':
                    trap = pd.concat([Traj_A, Traj_B], axis=0).median()
                    Traj_A = Traj_A - trap
                    Traj_B = Traj_B - trap
                if what_to_subtract.split(' ')[0] == 'traj':
                    Traj_A = Traj_A - Traj_A.median()
                    Traj_B = Traj_B - Traj_B.median()
            else:
                print('wrong what_to_subtract in generation_dataframe_creation')

            return Traj_A, Traj_B

        # creation of the dictionaries
        Both_dict = dict()
        A_dict = dict()
        B_dict = dict()
        log_Both_dict = dict()
        log_A_dict = dict()
        log_B_dict = dict()

        # loop over the amount of data in the subclass
        for experiment_number in range(data_len):
            # get the trajectory generation dataframe
            TrajA, TrajB = generation_dataframe_creation(dataID=experiment_number, _data=_data,
                                                              discretize_by=discretization_variable,
                                                              fit_the_length_at_birth=True)

            # apply the hard generation limits across all traces 
            TrajA = TrajA.iloc[start_index:end_index]
            TrajB = TrajB.iloc[start_index:end_index]

            log_TrajA = np.log(TrajA.iloc[start_index:end_index])
            log_TrajB = np.log(TrajB.iloc[start_index:end_index])

            TrajA, TrajB = minusing(what_to_subtract=what_to_subtract, Traj_A=TrajA, Traj_B=TrajB)
            log_TrajA, log_TrajB = minusing(what_to_subtract=what_to_subtract, Traj_A=log_TrajA, Traj_B=log_TrajB)

            # the general dataframe keys
            keyA = 'A_' + str(experiment_number)
            keyB = 'B_' + str(experiment_number)

            # saving the dataframes to the dictionary
            Both_dict.update({keyA: TrajA, keyB: TrajB})
            A_dict.update({str(experiment_number): TrajA})
            B_dict.update({str(experiment_number): TrajB})
            log_Both_dict.update({keyA: log_TrajA, keyB: log_TrajB})
            log_A_dict.update({str(experiment_number): log_TrajA})
            log_B_dict.update({str(experiment_number): log_TrajB})

        return Both_dict, A_dict, B_dict, log_Both_dict, log_A_dict, log_B_dict

    """ For input it gets a dictionary that contains dataframes, or an array that contains such dictionaries, then this function proceeds to cut the 
    first n generations from the dataframe(s) and returns the new dataframe or an array of the new dataframes in the same way they were given """
    def cut_the_dataframe(self, dicts, n=5):
        both_dict = dicts[0].copy()
        A_dict = dicts[1].copy()
        B_dict = dicts[2].copy()
        for key in both_dict.keys():
            both_dict[key] = both_dict[key].iloc[n:].reset_index(drop=True)
        for key in A_dict.keys():
            A_dict[key] = A_dict[key].iloc[n:].reset_index(drop=True)
        for key in B_dict.keys():
            B_dict[key] = B_dict[key].iloc[n:].reset_index(drop=True)

        return both_dict, A_dict, B_dict

        # if isinstance(dicts, dict):
        #     new_dict = dict(zip(dicts.keys(), dicts.values()))
        #     print(type(new_dict))
        #     exit()
        #     for key in new_dict.keys():
        #         new_dict[key] = new_dict[key].iloc[n:].reset_index(drop=True)
        #     return new_dict
        # elif isinstance(dicts, list):
        #     new_array = []
        #     for num_of_dicts in range(len(dicts)):
        #         if isinstance(dicts[num_of_dicts], dict):
        #             new_dict = dicts[num_of_dicts].copy()
        #             for key in new_dict.keys():
        #                 new_dict[key] = new_dict[key].iloc[n:].reset_index(drop=True)
        #             new_array.append(new_array)
        #         else:
        #             print('ERROR! The array needs to contain dictionaries! IN CUT_THE_DATAFRAME')
        #             pass
        #     what_to_return = iter(tuple(new_array))
        #     return what_to_return
        # else:
        #     print('ERROR! Need to input an array of dictionaries or a single dictionary, IN CUT_THE_DATAFRAME')
        #     pass

    # def give_the_trap_avg_for_control(key, sis_A, sis_B, non_sis_A, non_sis_B, start_index=None, end_index=None):
    #     # give the key of the reference of the dictionary, for example, "nonsis_A_57"
    #     if key.split('_')[0] == 'sis':
    #         ref_A = sis_A[key.split('_')[2]]
    #         ref_B = sis_B[key.split('_')[2]]
    #     else:
    #         ref_A = non_sis_A[key.split('_')[2]]
    #         ref_B = non_sis_B[key.split('_')[2]]
    #
    #     # decide what generations to use to determine the trap mean
    #     if start_index == None:
    #         if end_index == None:
    #             trap_mean = pd.concat([ref_A, ref_B], axis=0).reset_index(drop=True).mean()
    #         else:
    #             trap_mean = pd.concat([ref_A.iloc[:end_index], ref_B.iloc[:end_index]], axis=0).reset_index(drop=True).mean()
    #     else:
    #         if end_index == None:
    #             trap_mean = pd.concat([ref_A.iloc[start_index:], ref_B.iloc[start_index:]], axis=0).reset_index(drop=True).mean()
    #         else:
    #             trap_mean = pd.concat([ref_A.iloc[start_index:end_index], ref_B.iloc[start_index:end_index]], axis=0).reset_index(drop=True).mean()
    #
    #     return trap_mean
    #
    # def subtract_trap_averages(df_main, df_other, columns_names, start_index=None, end_index=None):
    #     df_new = pd.DataFrame(columns=columns_names)
    #     for col in columns_names:
    #         if start_index == None:
    #             if end_index == None:
    #                 trap_mean = pd.concat([df_main[columns_names], df_other[columns_names]], axis=0).reset_index(drop=True).mean()
    #                 df_new[col] = df_main[col] - trap_mean[col]
    #             else:
    #                 trap_mean = pd.concat([df_main[columns_names].iloc[:end_index], df_other[columns_names].iloc[:end_index]], axis=0).reset_index(
    #                     drop=True).mean()
    #                 df_new[col] = df_main[col].iloc[:end_index] - trap_mean[col]
    #         else:
    #             if end_index == None:
    #                 trap_mean = pd.concat([df_main[columns_names].iloc[start_index:], df_other[columns_names].iloc[start_index:]],
    #                                       axis=0).reset_index(drop=True).mean()
    #                 df_new[col] = df_main[col].iloc[start_index:] - trap_mean[col]
    #             else:
    #                 trap_mean = pd.concat([df_main[columns_names].iloc[start_index:end_index], df_other[columns_names].iloc[start_index:end_index]],
    #                                       axis=0).reset_index(drop=True).mean()
    #                 df_new[col] = df_main[col].iloc[start_index:end_index] - trap_mean[col]
    #
    #     return df_new
    #
    # def subtract_traj_averages(df, columns_names, start_index=None, end_index=None):
    #     df_new = pd.DataFrame(columns=columns_names)
    #     for col in columns_names:
    #         if start_index == None:
    #             if end_index == None:
    #                 df_new[col] = df[col] - df[col].mean()
    #             else:
    #                 df_new[col] = df[col].iloc[:end_index] - df[col].iloc[:end_index].mean()
    #         else:
    #             if end_index == None:
    #                 df_new[col] = df[col].iloc[start_index:] - df[col].iloc[start_index:].mean()
    #             else:
    #                 df_new[col] = df[col].iloc[start_index:end_index] - df[col].iloc[start_index:end_index].mean()
    #
    #     return df_new
    #
    # def subtract_trap_averages_control(df, columns_names, ref_key, sis_A, sis_B, non_sis_A, non_sis_B, start_index=None, end_index=None):
    #
    #     # we have to get the original trap mean
    #     trap_mean = give_the_trap_avg_for_control(ref_key, sis_A, sis_B, non_sis_A, non_sis_B, start_index, end_index)
    #
    #     df_new = pd.DataFrame(columns=columns_names)
    #     for col in columns_names:
    #         if start_index == None:
    #             if end_index == None:
    #                 df_new[col] = df[col] - trap_mean[col]
    #             else:
    #                 df_new[col] = df[col].iloc[:end_index] - trap_mean[col]
    #         else:
    #             if end_index == None:
    #                 df_new[col] = df[col].iloc[start_index:] - trap_mean[col]
    #             else:
    #                 df_new[col] = df[col].iloc[start_index:end_index] - trap_mean[col]
    #
    #     return df_new
    #
    # def subtract_global_averages(df, columns_names, pop_mean):
    #     df_new = pd.DataFrame(columns=columns_names)
    #     for col in columns_names:
    #         df_new[col] = df[col] - pop_mean[col]
    #
    #     return df_new

    """ plots the regression plots with the pearson correlation for two dataframes that have a certain relationship """
    def plot_relationship_correlations(self, df_1, df_2, df_1_variables, df_2_variables, x_labels, y_labels,
                                       limit_the_axes_func=limit_the_axes, corr_to_annotate_func=corrfunc):
        new_df_1 = df_1[df_1_variables].copy()
        new_df_1_vars = x_labels
        new_df_1 = new_df_1.rename(columns=dict(zip(df_1_variables, new_df_1_vars)))
        print('change of labels:\n', dict(zip(df_1_variables, new_df_1_vars)))

        new_df_2 = df_2[df_2_variables].copy()
        new_df_2_vars = y_labels
        new_df_2 = new_df_2.rename(columns=dict(zip(df_2_variables, new_df_2_vars)))
        print('change of labels:\n', dict(zip(df_2_variables, new_df_2_vars)))

        df = pd.concat([new_df_1[new_df_1_vars], new_df_2[new_df_2_vars]], axis=1)

        # plot the same-cell correlations
        g = sns.PairGrid(df, x_vars=new_df_1_vars, y_vars=new_df_2_vars)
        g = g.map(sns.regplot, line_kws={'color': 'orange'})
        g = g.map(limit_the_axes_func)
        g = g.map(corr_to_annotate_func)
        plt.show()
        save_fig = input("Do you want to save this graph? (Yes or No)")
        if save_fig == 'Yes':
            name = input("What name do you want to give this graph? (DO NOT put .png after)")
            g.savefig(name + ".png")
        elif save_fig == 'No':
            pass
        else:
            print('You inputed something other than Yes or No! Therefore graph is not saved.')

    """ Gets the trap and traj COV """

    def coeffs_of_variations(self, stats_df):
        coef_of_vars = stats_df.loc['std'] / stats_df.loc['mean']
        return coef_of_vars

    """ A function that goes into the init function and gets the corresponding protein raw data based on the list what_class """

    def get_protein_raw_data(self, what_class, **kwargs):
        _protein_data = list()
        _protein_origin = list()
        _protein_keylist = list()

        # Population will have what_class=['Sister', 'Nonsister'] because it includes both
        if 'Sister' in what_class:
            # where the excel files are
            _protein_sister = kwargs.get('infiles_sister_protein', [])

            # load first sheet of each Excel-File, fill internal data structure
            for filename in _protein_sister:
                try:
                    # creates a dataframe from the excel file
                    tmpdata = pd.read_excel(filename)
                except:
                    continue
                # the _data contains all dataframes from the excel files in the directory _infiles
                _protein_data.append(tmpdata)
                # the name of the excel file
                _protein_origin.append(filename)
                for k in tmpdata.keys():
                    if not str(k) in _protein_keylist:
                        # this list contains the column names of the excel file
                        _protein_keylist.append(str(k))

            class_name = 'Sister'

        if 'Nonsister' in what_class:
            _protein_nonsister = kwargs.get('infiles_nonsister_protein', [])
            # load first sheet of each Excel-File, fill internal data structure
            for filename in _protein_nonsister:
                try:
                    # creates a dataframe from the excel file
                    tmpdata = pd.read_excel(filename)
                except:
                    continue
                # the _data contains all dataframes from the excel files in the directory _infiles
                _protein_data.append(tmpdata)
                # the name of the excel file
                _protein_origin.append(filename)
                for k in tmpdata.keys():
                    if not str(k) in _protein_keylist:
                        # this list contains the column names of the excel file
                        _protein_keylist.append(str(k))

            if 'Sister' in what_class:
                class_name = 'Population'
            else:
                class_name = 'Nonsister'

        # use this for the loops later on
        _protein_data_len = len(_protein_data)

        # there's no point in not having data ...
        # ... or something went wrong. rather stop here
        if not _protein_data_len > 0:
            raise IOError('no data loaded')

        if class_name == 'Population':
            # Here since we are only talking about the population we don't care about A and B traces.
            # The datasets are mixed anyways so the A/B difference doesn't tell us anything.
            _all_raw_protein_data_dict, _, _ = self.create_dictionaries_of_mean_fluor(protein_data_len=_protein_data_len, protein_data=_protein_data)
            print('Got the _all_raw_protein_data_dict attribute in ' + class_name)

            return _protein_data, _protein_data_len, _protein_keylist, _protein_origin, _all_raw_protein_data_dict

        else:
            # Here we are specifically interested in the A/B traces
            _all_raw_protein_data_dict, A_raw_protein_data_dict, B_raw_protein_data_dict = self.create_dictionaries_of_mean_fluor(
                protein_data_len=_protein_data_len, protein_data=_protein_data)
            print('Got the _all_raw_protein_data_dict and A/B_raw_protein_data_dict attribute in ' + class_name)

            return _protein_data, _protein_data_len, _protein_keylist, _protein_origin, _all_raw_protein_data_dict, A_raw_protein_data_dict, \
                   B_raw_protein_data_dict

    """ A function that goes into the init function and gets the corresponding protein raw data based on the list what_class """

    def gen_dicts_and_class_stats(self, what_class, **kwargs):

        # The observable we will use to see when the cell divided and what the cycle data is based on (can be either 'length',
        # 'cellarea', or 'fluorescence')
        _discretization_variable = kwargs.get('discretization_variable', [])

        number_of_gens_to_cut_from_start = kwargs.get('number_of_gens_to_cut_from_start', 5)

        what_to_subtract = kwargs.get('what_to_subtract', None)
        start_index = kwargs.get('start_index', None)
        end_index = kwargs.get('end_index', None)

        # where to put the raw data
        _data = list()
        _origin = list()
        _keylist = list()

        # Population will have what_class=['Sister', 'Nonsister'] because it includes both
        if 'Sister' in what_class:
            # where the excel files are
            _sister = kwargs.get('infiles_sister', [])

            # load first sheet of each Excel-File, fill internal data structure
            for filename in _sister:
                try:
                    # creates a dataframe from the excel file
                    tmpdata = pd.read_excel(filename)
                except:
                    continue
                # the _data contains all dataframes from the excel files in the directory _infiles
                _data.append(tmpdata)
                # the name of the excel file
                _origin.append(filename)
                for k in tmpdata.keys():
                    if not str(k) in _keylist:
                        # this list contains the column names of the excel file
                        _keylist.append(str(k))

            # If it is sister then this will stay, if it Population then we have the if-else statement below
            class_name = 'Sister'

        if 'Nonsister' in what_class:
            _nonsister = kwargs.get('infiles_nonsister', [])
            # load first sheet of each Excel-File, fill internal data structure
            for filename in _nonsister:
                try:
                    # creates a dataframe from the excel file
                    tmpdata = pd.read_excel(filename)
                except:
                    continue
                # the _data contains all dataframes from the excel files in the directory _infiles
                _data.append(tmpdata)
                # the name of the excel file
                _origin.append(filename)
                for k in tmpdata.keys():
                    if not str(k) in _keylist:
                        # this list contains the column names of the excel file
                        _keylist.append(str(k))

            # the if-else statement mentioned above, decides if it is the Nonsister class or Population
            if 'Sister' in what_class:
                class_name = 'Population'
            else:
                class_name = 'Nonsister'

        # use this for the loops later on
        _data_len = len(_data)

        # there's no point in not having data ...
        # ... or something went wrong. rather stop here
        if not _data_len > 0:
            raise IOError('no data loaded')

        if class_name == 'Population':
            # Here since we are only talking about the population we don't care about A and B traces.
            # The datasets are mixed anyways so the A/B difference doesn't tell us anything.
            _all_data_dict, _, _, log_all_data_dict, _, _ = self.create_dictionaries_of_traces(data_len=_data_len, discretization_variable=_discretization_variable,
                                                                      _keylist=_keylist, _data=_data, what_to_subtract=what_to_subtract, start_index=start_index, end_index=end_index)
            print('Got the (log)_all_data_dict attributes in Population')

            # Get the all bacteria dataset
            all_bacteria = self.create_all_bacteria_dict(_all_data_dict)
            print('got all_bacteria')
            log_all_bacteria = self.create_all_bacteria_dict(log_all_data_dict)
            print('got log_all_bacteria')

            return _data, _data_len, _keylist, _origin, _all_data_dict, log_all_data_dict, all_bacteria, log_all_bacteria
            # return _data, _data_len, _keylist, _origin, _all_data_dict, pop_stats, coeffs_of_vars, all_bacteria
        else:
            # For the Env_Sister subclass put in True, else it will be a normal Sister class
            cut_out_genetic_contribution = kwargs.get('cut_out_genetic_contribution', False)

            Both_dict, A_dict, B_dict, log_Both_dict, log_A_dict, log_B_dict = self.create_dictionaries_of_traces(data_len=_data_len, discretization_variable=_discretization_variable,
                                                                           _keylist=_keylist, _data=_data, what_to_subtract=what_to_subtract, start_index=start_index, end_index=end_index)
            print('Got the Both_dict, A_dict, and B_dict attributes for the ' + class_name + ' class')

            # if it is the Env_Sister class
            if cut_out_genetic_contribution:
                Both_dict, A_dict, B_dict = self.cut_the_dataframe(dicts=[Both_dict, A_dict, B_dict], n=number_of_gens_to_cut_from_start)
                print('The Sister dataframes in the dictionaries were cut, for the Env_Sister Class')

            A_stats_dict, B_stats_dict, trap_stats_dict = self.trap_and_traj_variable_statistics(A_dict=A_dict, B_dict=B_dict)
            print('Got the A_stats_dict, B_stats_dict, and trap_stats_dict attributes for the ' + class_name + ' class')

            Trap_coeffs_of_vars_dict = dict()
            # Get the coefficients of variation
            # FIXME: GET THIS OUT OF HEREEEEE
            for key, val in zip(trap_stats_dict.keys(), trap_stats_dict.values()):
                Trap_coeffs_of_vars = self.coeffs_of_variations(stats_df=val)
                Trap_coeffs_of_vars_dict.update({key: Trap_coeffs_of_vars})

            A_coeffs_of_vars_dict = dict()
            # Get the coefficients of variation
            # FIXME: GET THIS OUT OF HEREEEEE
            for key, val in zip(A_stats_dict.keys(), A_stats_dict.values()):
                A_coeffs_of_vars = self.coeffs_of_variations(stats_df=val)
                A_coeffs_of_vars_dict.update({key: A_coeffs_of_vars})

            B_coeffs_of_vars_dict = dict()
            # Get the coefficients of variation
            # FIXME: GET THIS OUT OF HEREEEEE
            for key, val in zip(B_stats_dict.keys(), B_stats_dict.values()):
                B_coeffs_of_vars = self.coeffs_of_variations(stats_df=val)
                B_coeffs_of_vars_dict.update({key: B_coeffs_of_vars})
            print('Got the coefficients of variation for A/B traces and traps for the ' + class_name + ' class')

            return _data, _data_len, _keylist, _origin, A_dict, B_dict, Both_dict, A_stats_dict, B_stats_dict, trap_stats_dict, \
                   Trap_coeffs_of_vars_dict, A_coeffs_of_vars_dict, B_coeffs_of_vars_dict

    """ For the same-cell statistics """

    def create_all_bacteria_dict(self, all_data_dict):
        # colsss = self._relationship_dfs_variable_names + ['trap_ID']
        # all_bacteria = pd.DataFrame(columns=colsss)
        all_bacteria = pd.DataFrame(columns=self._variable_names + ['trap_ID'])
        for keyA, keyB in zip(list(all_data_dict.keys())[:-1:2], list(all_data_dict.keys())[1::2]):
            # # concatenate both the A and B trajectory in that order
            # both_traj_df = pd.concat([all_data_dict[keyA].copy(), all_data_dict[keyB].copy()]).reset_index(drop=True)
            final_df = pd.concat([all_data_dict[keyA].copy(), all_data_dict[keyB].copy()]).reset_index(drop=True)

            # # trap and traj means dataframe creation
            # trap_means = pd.DataFrame(columns=self._variable_names)
            # traj_means = pd.DataFrame(columns=self._variable_names)
            # log_trap_means = pd.DataFrame(columns=self._variable_names)
            # log_traj_means = pd.DataFrame(columns=self._variable_names)

            # if len(np.where(both_traj_df <= 0)[0]) > 0:
            #     print_full_dataframe(both_traj_df)
            #     exit()

            # # loop over all the cells in the trap
            # for ind in range(len(both_traj_df)):
            #
            #     # # save the trap means for every cell
            #     # trap_means = trap_means.append(both_traj_df.mean(), ignore_index=True)
            #     # log_trap_means = log_trap_means.append(np.log(both_traj_df).mean(), ignore_index=True)
            #
            #     # # if the row belongs to the A trace, save the traj means for every cell
            #     # if ind < len(all_data_dict[keyA]):
            #     #     traj_means = traj_means.append(all_data_dict[keyA].mean(), ignore_index=True)
            #     #     log_traj_means = log_traj_means.append(np.log(all_data_dict[keyA]).mean(), ignore_index=True)
            #
            #     # # if the row belongs to the B trace, save the traj means for every cell
            #     # else:
            #     #     traj_means = traj_means.append(all_data_dict[keyB].mean(), ignore_index=True)
            #     #     log_traj_means = log_traj_means.append(np.log(all_data_dict[keyB]).mean(), ignore_index=True)

            # # rename the columns of the trajectory and trap means, concatenate them together
            # # and then concatenate that one to the df with all the variables
            # traj_means = traj_means.rename(columns=dict(zip(self._variable_names, self._traj_variable_names)))
            # trap_means = trap_means.rename(columns=dict(zip(self._variable_names, self._trap_variable_names)))
            # log_traj_means = log_traj_means.rename(columns=dict(zip(self._variable_names, ['log_' + name for name in self._traj_variable_names])))
            # log_trap_means = log_trap_means.rename(columns=dict(zip(self._variable_names, ['log_' + name for name in self._trap_variable_names])))
            # means_df = pd.concat([trap_means, traj_means, log_trap_means, log_traj_means], axis=1)
            # final_df = pd.concat([both_traj_df, means_df], axis=1)

            if keyA.split('_')[1] != keyB.split('_')[1]:
                print('different keys in all_bacteria creation')
                exit()

            final_df['trap_ID'] = pd.Series(keyA.split('_')[1]).repeat(len(final_df)).reset_index(drop=True)

            all_bacteria = pd.concat([all_bacteria, final_df], axis=0, join='inner').reset_index(drop=True)

        return all_bacteria

    """ logs the dictionaries in whatever way we want """

    def log_the_gen_dict(self, **kwargs):
        dictionary = kwargs.get('dictionary', [])
        variables_to_log = kwargs.get('variables_to_log', [])

        # the new dictionary with the new dataframes
        new_dictionary = dict()

        # loop over all trajectories in S and NS datasets
        for key in dictionary.keys():
            # to have everything else the same
            dictionary_key = dictionary[key].copy()

            for var in variables_to_log:
                # normalizing the two length observations for all trajectories in S and NS datasets
                dictionary_key[var] = np.log(dictionary[key][var])

            # save the new dataframe to the new dictionary
            new_dictionary.update({key: dictionary_key})

        return new_dictionary

    """ Creates a dataframe that contains the population mean and std div (ONLY for Population subclass) """

    def population_variable_statistics(self, _all_data_dict):

        # The dataframe that will have the means and std divs of all bacteria recorded, both Sister and Nonsister datasets combined
        stats = pd.DataFrame(columns=self._variable_names, index=['mean', 'std'])

        # loop over variables
        for variable in self._variable_names:
            # concatenate all columns of this variable from all the bacteria observed
            stats[variable].loc['mean'] = np.mean(np.concatenate([_all_data_dict[dataID][variable] for dataID in _all_data_dict.keys()]))
            stats[variable].loc['std'] = np.std(np.concatenate([_all_data_dict[dataID][variable] for dataID in _all_data_dict.keys()]),
                                                ddof=1)

        return stats

    # """ The autocorrelation function for one trace """
    # def cross_correlation(self, vec_1, label_1, vec_2, label_2, plot, title):
    #
    #     min_num = min(len(vec_1), len(vec_2))
    #     vec_1 = vec_1.copy().iloc[:min_num]
    #     vec_2 = vec_2.copy().iloc[:min_num]
    #     corr_1 = np.array([stats.pearsonr(vec_1, vec_2)[0]] +
    #                       [stats.pearsonr(vec_1[t:], vec_2[:-t])[0] for t in np.arange(1, .5*min_num, dtype=np.int32)])
    #     corr_2 = np.array([stats.pearsonr(vec_1, vec_2)[0]] +
    #                       [stats.pearsonr(vec_2[t:], vec_1[:-t])[0] for t in np.arange(1, .5 * min_num, dtype=np.int32)])
    #     if plot:
    #         sns.set_style('darkgrid')
    #         fig, (ax_signals, ax_corr_1, ax_corr_2) = plt.subplots(3, 1)
    #         # fig.tight_layout()
    #         sns.lineplot(x=np.arange(1, len(vec_1)+1)*3, y=vec_1, label=label_1, ax=ax_signals, color='blue')
    #         sns.lineplot(x=np.arange(1, len(vec_2)+1)*3, y=vec_2, label=label_2, ax=ax_signals, color='orange')
    #
    #         ax_signals.set_ylabel('Fluorescence')
    #         # ax_signals.set_xlabel('Mins')
    #         ax_signals.axhline(y=np.mean(vec_1), ls='--', color='blue')
    #         ax_signals.axhline(y=np.mean(vec_2), ls='--', color='orange')
    #         sns.lineplot(x=np.arange(1, len(corr_1)+1)*3, y=corr_1, ax=ax_corr_1)
    #         sns.lineplot(x=np.arange(1, len(corr_2)+1)*3, y=corr_2, ax=ax_corr_2)
    #         # ax_corr_1.set_title('Cross-correlation (blue forward)')
    #         ax_corr_1.set_xlabel('t: Point Displacement')
    #         ax_corr_1.set_ylabel(r'$\rho(x[t:], y[:-t])$')
    #         # ax_corr_2.set_title('Cross-correlation (orange forward)')
    #         ax_corr_2.set_xlabel('t: Point Displacement')
    #         ax_corr_2.set_ylabel(r'$\rho(y[t:], x[:-t])$')
    #         ax_corr_1.axhline(y=0, ls='--', color='black')
    #         ax_corr_2.axhline(y=0, ls='--', color='black')
    #         fig.tight_layout()
    #         fig.suptitle(title)
    #         fig.savefig('Cross-Correlations '+title+' '+label_1, dpi=400)
    #         plt.close(fig)
    #
    #         # g = sns.FacetGrid(subplot_kws={'nrows': 3, 'ncols': 1, 'sharex': False, 'sharey': True})
    #         # g.map(sns.distplot, "total_bill", hist=False, rug=True)
    #     # return corr_1, corr_2


    # access single dataframe by its ID
    def __getitem__(self, key):
        return self._data[key]

    # should not allow accessing internal variables in other ways than funneling through this here
    def __getattr__(self, key):
        if key == "filenames":
            return self._dataorigin
        elif key == "keylist":
            return self._keylist
        elif key == "keylist_stripped":
            return list(set([s.strip('AB ') for s in self._keylist]))
        elif key == 'timestep':
            ts = self[0]['timeA'][1] - self[0]['timeA'][0]
            for dataID in range(self._all_data_len):
                dt = np.diff(self._data[dataID]['timeA'])
                if ts > np.min(dt): ts = np.min(dt)
            return ts

    # data will be processes as loop over the class instance
    # 'debugmode' only returns a single item (the first one)
    def __iter__(self):
        dataIDs = np.arange(self._all_data_len, dtype=int)
        for dataID, origin, data in zip(dataIDs, self._dataorigin, self._data):
            yield dataID, origin, data

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        return vars(self).update(state)


class Global_Mean(SisterCellData):
    def __init__(self, **kwargs):

        self.Population = Population(**kwargs)
        self.Sister = Sister(**kwargs)
        # self.Nonsister = Nonsister.__init__(self, **kwargs)
        # self.Control = Control.__init__(self, **kwargs)


class Population(SisterCellData):

    """ Here we create the mother and daughter dataframe that contains all the instances of mother and daughter across Sister and NS datasets.
        This is only meant for the Population subclass because we are not differentiating between A/B.
        We can choose between the normalized lengths dictionary or the units one. """

    def intergenerational_dataframe_creations(self, all_data_dict, how_many_separations):

        # These are where we will put the older and newer generations respectively
        mother_dfs = []
        daughter_dfs = []

        # 1 separation is mother daughter, 2 is grandmother granddaughter, etc...
        for separation in range(1, how_many_separations+1):
            # the dataframe for the older and the younger generations
            older_df = pd.DataFrame(columns=self._relationship_dfs_variable_names)
            younger_df = pd.DataFrame(columns=self._relationship_dfs_variable_names)

            # loop over every key in both S and NS datasets
            for keyA, keyB in zip(list(all_data_dict.keys())[:-1:2], list(all_data_dict.keys())[1::2]):

                # concatenate both the A and B trajectory in that order
                both_traj_df = pd.concat([all_data_dict[keyA].copy(), all_data_dict[keyB].copy()]).reset_index(drop=True)

                # trap and traj means dataframe creation
                trap_means = pd.DataFrame(columns=self._variable_names)
                traj_means = pd.DataFrame(columns=self._variable_names)
                log_trap_means = pd.DataFrame(columns=self._variable_names)
                log_traj_means = pd.DataFrame(columns=self._variable_names)

                if len(np.where(both_traj_df <= 0)[0]) > 0:
                    print_full_dataframe(both_traj_df)
                    exit()

                # loop over all the cells in the trap
                for ind in range(len(both_traj_df)):

                    # save the trap means for every cell
                    trap_means = trap_means.append(both_traj_df.mean(), ignore_index=True)
                    log_trap_means = log_trap_means.append(np.log(both_traj_df).mean(), ignore_index=True)

                    # if the row belongs to the A trace, save the traj means for every cell
                    if ind < len(all_data_dict[keyA]):
                        traj_means = traj_means.append(all_data_dict[keyA].mean(), ignore_index=True)
                        log_traj_means = log_traj_means.append(np.log(all_data_dict[keyA]).mean(), ignore_index=True)

                    # if the row belongs to the B trace, save the traj means for every cell
                    else:
                        traj_means = traj_means.append(all_data_dict[keyB].mean(), ignore_index=True)
                        log_traj_means = log_traj_means.append(np.log(all_data_dict[keyB]).mean(), ignore_index=True)

                # rename the columns of the trajectory and trap means, concatenate them together
                # and then concatenate that one to the df with all the variables
                traj_means = traj_means.rename(columns=dict(zip(self._variable_names, self._traj_variable_names)))
                trap_means = trap_means.rename(columns=dict(zip(self._variable_names, self._trap_variable_names)))
                log_traj_means = log_traj_means.rename(columns=dict(zip(self._variable_names, ['log_'+name for name in self._traj_variable_names])))
                log_trap_means = log_trap_means.rename(columns=dict(zip(self._variable_names, ['log_'+name for name in self._trap_variable_names])))
                means_df = pd.concat([trap_means, traj_means, log_trap_means, log_traj_means], axis=1)
                final_df = pd.concat([both_traj_df, means_df], axis=1)

                # decides which generation to put so we don't have any conflicting or non-true inter-generational pairs
                older_mask = [ind for ind in range(len(all_data_dict[keyA])-separation)] + \
                             [len(all_data_dict[keyA])+ind for ind in range(len(all_data_dict[keyB])-separation)]
                younger_mask = [separation + ind for ind in range(len(all_data_dict[keyA]) - separation)] + \
                               [separation + len(all_data_dict[keyA]) + ind for ind in range(len(all_data_dict[keyB]) - separation)]

                # add this trap's mother and daughter cells to the df
                # print(len(final_df.iloc[older_mask].columns), len(older_df.columns))
                # print(len([col1 for col1 in final_df.iloc[older_mask].columns if (col1 in older_df.columns)]))
                # print(len([col1 for col1 in older_df.columns if (col1 in final_df.iloc[older_mask].columns)]))
                # print(final_df.iloc[older_mask].columns, older_df.columns)
                # print([col1 for col1 in final_df.iloc[older_mask].columns for col2 in older_df.columns if col1 in older_df.columns and col2 in final_df.iloc[older_mask].columns])
                # print(len(final_df.iloc[younger_mask].columns))
                # if final_df.iloc[older_mask].columns != older_df.columns:
                #     print('older dont match')
                # if final_df.iloc[younger_mask].columns != younger_df.columns:
                #     print('younger dont match')
                older_df = pd.concat([older_df, final_df.iloc[older_mask]], axis=0, join='inner').reset_index(drop=True)
                younger_df = pd.concat([younger_df, final_df.iloc[younger_mask]], axis=0, join='inner').reset_index(drop=True)

                # print(len(older_df.columns), len(younger_df.columns))

            # append them to the list that holds all the dataframes
            mother_dfs.append(older_df)
            daughter_dfs.append(younger_df)

        return mother_dfs, daughter_dfs

    """ Creates pairs of dataframes of intergenerational relations from all the A/B pairs we find in the subclass's A_dict
            and B_dict. This method is only meant for Sister, Nonsister and Control subclasses; NOT Population. """

    def intragenerational_dataframe_creations(self, A_dict, B_dict, how_many_cousins_and_sisters):

        A_df_array = []
        B_df_array = []

        for generation in range(how_many_cousins_and_sisters + 1):
            A_df = pd.DataFrame(columns=self._relationship_dfs_variable_names)
            B_df = pd.DataFrame(columns=self._relationship_dfs_variable_names)

            # Because it is not a given that all the experiments done will have a this many generations recorded
            A_keys_with_this_length = [keyA for keyA, keyB in zip(A_dict.keys(), B_dict.keys()) if
                                       min(len(A_dict[keyA]), len(B_dict[keyB])) > generation + 1]
            B_keys_with_this_length = [keyB for keyA, keyB in zip(A_dict.keys(), B_dict.keys()) if
                                       min(len(A_dict[keyA]), len(B_dict[keyB])) > generation + 1]

            # looping over all pairs recorded, ie. all traps/experiments
            # for keyA, keyB in zip(A_dict.keys(), B_dict.keys()):
            for keyA, keyB in zip(A_keys_with_this_length, B_keys_with_this_length):
                # concatenate both the A and B trajectory in that order
                both_traj_df = pd.concat([A_dict[keyA], B_dict[keyB]]).reset_index(drop=True)

                # Get the trap and trajectory means and convert them from a Series into a dataframe with their respective columns
                trap_mean = both_traj_df.mean().to_frame().T.rename(columns=dict(zip(self._variable_names, self._trap_variable_names)))
                A_traj_mean = A_dict[keyA].mean().to_frame().T.rename(columns=dict(zip(self._variable_names, self._traj_variable_names)))
                B_traj_mean = B_dict[keyB].mean().to_frame().T.rename(columns=dict(zip(self._variable_names, self._traj_variable_names)))

                log_trap_mean = both_traj_df.mean().to_frame().T.rename(
                    columns=dict(zip(self._variable_names, ['log_' + name for name in self._trap_variable_names])))
                log_A_traj_mean = A_dict[keyA].mean().to_frame().T.rename(
                    columns=dict(zip(self._variable_names, ['log_' + name for name in self._traj_variable_names])))
                log_B_traj_mean = B_dict[keyB].mean().to_frame().T.rename(
                    columns=dict(zip(self._variable_names, ['log_' + name for name in self._traj_variable_names])))

                # concatenate the trap and trajectory means for both A/B trajectories respectively
                A_means = pd.concat([trap_mean, A_traj_mean, log_trap_mean, log_A_traj_mean], axis=1)
                B_means = pd.concat([trap_mean, B_traj_mean, log_trap_mean, log_B_traj_mean], axis=1)

                # concatenate the data to the trap and trajectory means
                what_to_add_to_A_df = pd.concat([A_dict[keyA].iloc[generation].to_frame().T.reset_index(drop=True), A_means], axis=1)
                what_to_add_to_B_df = pd.concat([B_dict[keyB].iloc[generation].to_frame().T.reset_index(drop=True), B_means], axis=1)

                # add the data, trap means and traj means to the dataframe that collects them for all traps/experiments
                A_df = pd.concat([A_df, what_to_add_to_A_df], axis=0)
                B_df = pd.concat([B_df, what_to_add_to_B_df], axis=0)

            # reset the index because it is convinient and they were all added with an index 0 since we are comparing sisters
            A_df = A_df.reset_index(drop=True)
            B_df = B_df.reset_index(drop=True)

            # add it to the array that contains the sisters, first cousins, second cousins, etc...
            A_df_array.append(A_df)
            B_df_array.append(B_df)

        return A_df_array, B_df_array

    """ Has the histograms on the diagonals and the regression plots with the pearson correlation on the off diagonals """
    def plot_same_cell_correlations(self, df, variables, labels, limit_the_axes_func=limit_the_axes, corr_to_annotate_func=corrfunc):
        new_df = df[variables].copy()
        new_vars = labels
        new_df = new_df.rename(columns=dict(zip(variables, new_vars)))
        print('change of labels:\n', dict(zip(variables, new_vars)))
        # plot the same-cell correlations
        g = sns.PairGrid(new_df, x_vars=new_vars, y_vars=new_vars, diag_sharey=False)
        g = g.map_diag(sns.distplot)
        g = g.map_offdiag(sns.regplot, line_kws={'color': 'orange'})
        g = g.map_offdiag(limit_the_axes_func)
        g = g.map_offdiag(corr_to_annotate_func)
        plt.show()
        save_fig = input("Do you want to save this graph? (Yes or No)")
        if save_fig == 'Yes':
            name = input("What name do you want to give this graph? (DO NOT put .png after)")
            g.savefig(name + ".png")
        elif save_fig == 'No':
            pass
        else:
            print('You inputed something other than Yes or No! Therefore graph is not saved.')


    def __init__(self, **kwargs):

        # get the generation data
        self._data, self._data_len, self._keylist, self._origin, self._all_data_dict, self.log_all_data_dict, self.all_bacteria, self.log_all_bacteria = \
            self.gen_dicts_and_class_stats(what_class=['Sister', 'Nonsister'], **kwargs)

        print('got the generation data')

        if (kwargs.get('what_to_subtract').split(' ')[0] == 'global') or (kwargs.get('what_to_subtract') == None):
            print('what_to_subtract is global or None')
            # Get the population mean and variance of all variables
            # FIXME: Get this out of here!!!!!
            self.pop_stats = self.population_variable_statistics(_all_data_dict=self._all_data_dict)
            print('Got the pop stats attributes in Population')

            # Get the coefficients of variation
            # FIXME: Get this out of here!!!!!
            self.coeffs_of_vars = self.coeffs_of_variations(stats_df=self.pop_stats)
            print('Got the coeffs_of_vars attributes in Population')

        print('got in front of bare_minimum_pop')

        # to use for the subclasses so they don't have to calculate unnecessary data
        bare_minimum_pop = kwargs.get('bare_minimum_pop', [])

        # subclasses will put bare_minimum=True
        if bare_minimum_pop:
            pass
        # the instance of the Population class won't put anything
        else:
            # get the protein data
            self._protein_data, self._protein_data_len, self._protein_keylist, self._protein_origin, self._all_raw_protein_data_dict = self.get_protein_raw_data(what_class=['Sister', 'Nonsister'],
                **kwargs)

            # log some variables
            # self._log_all_data_dict = self.log_the_gen_dict(dictionary=self._all_data_dict,
            #                                                                variables_to_log=['length_birth', 'length_final', 'division_ratio'])
            # print('Got the _log_all_data_dict attribute in Population')

            # # Get the mother daughter DFs
            # self.mother_df, self.daughter_df, self.grand_mother_df, self.grand_daughter_df, self.great_grand_mother_df, self.great_grand_daughter_df = \
            #     self.intergenerational_dataframe_creations(all_data_dict=self._normalized_lengths_all_data_dict)
            # print('Got the mother and daughter dataframes in Population')

            # Get the mother daughter arrays of the dfs
            self.mother_dfs, self.daughter_dfs = \
                self.intergenerational_dataframe_creations(all_data_dict=self._all_data_dict, how_many_separations=6)
            print('Got the mother array and daughter array dataframes in Population')

            # contains all the attributes and methods
            self.organizational_table = self.table_with_attributes_and_available_methods()
            print('Got the organizational table for Population')

class Sister(Population):

    def __init__(self, **kwargs):
        # initiate the Population class that will pass down only the bare_minimum, ie. population dictionaries and statistics
        Population.__init__(self, **kwargs)

        # to use for the subclasses so they don't have to calculate unnecessary data
        bare_minimum_sis = kwargs.get('bare_minimum_sis', [])

        print('got this far!')

        # get the protein data
        self._protein_data, self._protein_data_len, self._protein_keylist, self._protein_origin, self._dataset_raw_protein_data_dict, \
        self.A_raw_protein_data_dict, self.B_raw_protein_data_dict = self.get_protein_raw_data(what_class=['Sister'], **kwargs)

        print('passed Sis protein data!')
        # get the generation data
        self._data, self._data_len, self._keylist, self._origin, self.A_dict, self.B_dict, self.Both_dict, self.A_stats_dict, self.B_stats_dict, \
        self.trap_stats_dict, self.Trap_coeffs_of_vars_dict, self.A_coeffs_of_vars_dict, self.B_coeffs_of_vars_dict = \
            self.gen_dicts_and_class_stats(what_class=['Sister'], **kwargs)

        # subclass Control will put bare_minimum=True
        if bare_minimum_sis:
            pass
        # the instance of the Sister class won't put anything
        else:
            # log some variables in the gen_dicts
            self._log_A_dict = self.log_the_gen_dict(dictionary=self.A_dict, variables_to_log=['length_birth', 'length_final',
                                                                                                         'division_ratio'])
            self._log_B_dict = self.log_the_gen_dict(dictionary=self.B_dict, variables_to_log=['length_birth', 'length_final',
                                                                                                           'division_ratio'])
            self._log_Both_dict = self.log_the_gen_dict(dictionary=self.Both_dict, variables_to_log=['length_birth', 'length_final',
                                                                                                           'division_ratio'])
            print('Got the _log_A/B/Both_dict attributes for Sister')

            # Get the relationship dataframes, made with the normalized lengths
            self.A_intra_gen_bacteria, self.B_intra_gen_bacteria = self.intragenerational_dataframe_creations(
                A_dict=self.A_dict, B_dict=self.B_dict, how_many_cousins_and_sisters=6)
            print('got the intragenerational_dataframes for Sister')

            # contains all the attributes and methods
            self.organizational_table = self.table_with_attributes_and_available_methods()
            print('got the organizational table for Sister')
            

class Env_Sister(Sister):
    def __init__(self, **kwargs):

        # This is Environemnt sisters dataset so we cut out the first 5
        # Run the sisters with shortened dictionaries
        Sister.__init__(self, cut_out_genetic_contribution=True, **kwargs)

    pass


class Nonsister(Population):

    def __init__(self, **kwargs):
        # initiate the Population class that will pass down only the bare_minimum, ie. population dictionaries and statistics
        Population.__init__(self, **kwargs)

        # get the protein data
        self._protein_data, self._protein_data_len, self._protein_keylist, self._protein_origin, self._dataset_raw_protein_data_dict, \
        self.A_raw_protein_data_dict, self.B_raw_protein_data_dict = self.get_protein_raw_data(what_class=['Nonsister'], **kwargs)

        # get the generation data
        self._data, self._data_len, self._keylist, self._origin, self.A_dict, self.B_dict, self.Both_dict, self.A_stats_dict, self.B_stats_dict, \
        self.trap_stats_dict, self.Trap_coeffs_of_vars_dict, self.A_coeffs_of_vars_dict, self.B_coeffs_of_vars_dict = \
            self.gen_dicts_and_class_stats(what_class=['Nonsister'], **kwargs)

        # to use for the subclasses so they don't have to calculate unnecessary data
        bare_minimum_non_sis = kwargs.get('bare_minimum_non_sis', [])

        # subclass Control will put bare_minimum=True
        if bare_minimum_non_sis:
            pass
        # the instance of the Nonsister class won't put anything
        else:
            # normalize the lengths
            # log some variables in the gen_dicts
            self._log_A_dict = self.log_the_gen_dict(dictionary=self.A_dict, variables_to_log=['length_birth', 'length_final',
                                                                                               'division_ratio'])
            self._log_B_dict = self.log_the_gen_dict(dictionary=self.B_dict, variables_to_log=['length_birth', 'length_final',
                                                                                               'division_ratio'])
            self._log_Both_dict = self.log_the_gen_dict(dictionary=self.Both_dict, variables_to_log=['length_birth', 'length_final',
                                                                                                     'division_ratio'])
            print('Got the _log_A/B/Both_dict attributes for Nonsister')

            # # Get the relationship dataframes
            # self.sister_A_df, self.sister_B_df, self.first_cousin_A_df, self.first_cousin_B_df, self.second_cousin_A_df, \
            # self.second_cousin_B_df, self.third_cousin_A_df, self.third_cousin_B_df = self.intragenerational_dataframe_creations(
            #     A_dict=self._normalized_lengths_A_dict, B_dict=self._normalized_lengths_B_dict)
            # print('got the intragenerational_dataframes for Nonsister')

            # Get the relationship dataframes, made with the normalized lengths
            self.A_intra_gen_bacteria, self.B_intra_gen_bacteria = self.intragenerational_dataframe_creations(
                A_dict=self.A_dict, B_dict=self.B_dict, how_many_cousins_and_sisters=6)
            print('got the intragenerational_dataframes for Sister')

            # contains all the attributes and methods
            self.organizational_table = self.table_with_attributes_and_available_methods()
            print('got the organizational table for Nonsister')


class Env_Nonsister(Nonsister):

    def __init__(self, **kwargs):
        # This is Environemnt sisters dataset so we cut out the first 5
        # Run the sisters with shortened dictionaries
        Nonsister.__init__(self, cut_out_genetic_contribution=True, number_of_gens_to_cut_from_start=1, **kwargs)

    pass


class Control(Sister, Nonsister):
    # """ Creating the Control dictionaries from the Sisters and Nonsisters dictionaries, Trace A will come from Sisters and Trace B will come from
    # Nonsisters , returns A_dict, B_dict, Both_dict, reference_A_dict, reference_B_dict """
    # def get_Control_A_B_and_Both_dicts(self, sis_Both, non_sis_Both, **kwargs):
    #     # if we want some debugging data
    #     debug = kwargs.get('debug', [])
    #
    #     # array of the key, dataframe and length of all bacteria in sis and nonsis dataset, and their concatenation
    #     sis_array = np.array([['sis_'+key, df_value, len(df_value)] for key, df_value in sis_Both.copy().items()])
    #     non_sis_array = np.array([['non_sis_'+key, df_value, len(df_value)] for key, df_value in non_sis_Both.copy().items()])
    #
    #     # to get the intersection of on which generations a cell finishes on both Sisters and Nonsisters datasets
    #     sis_lengths_of_gens_seen = np.unique(sis_array[:, 2])
    #     non_sis_lengths_of_gens_seen = np.unique(non_sis_array[:, 2])
    #     overlapping_lengths = np.intersect1d(sis_lengths_of_gens_seen, non_sis_lengths_of_gens_seen)
    #
    #     # to show that these "generation lengths" span 95% of all generation lengths of Sisters and Nonsisters datasets
    #     if debug == True:
    #         sis_keys_in_overlapping_lengths = [key for key, length in zip(sis_array[:, 0], sis_array[:, 2]) if length in overlapping_lengths]
    #         print('sis_keys_in_overlapping_lengths\n', sis_keys_in_overlapping_lengths)
    #         sis_keys_in_overlapping_lengths_unique = np.unique([[int(s) for s in sis_keys_in_overlapping_lengths[ind].split('_') if s.isdigit()] for
    #                                                             ind in range(len(sis_keys_in_overlapping_lengths))])
    #         print('sis_keys_in_overlapping_lengths_unique\n', sis_keys_in_overlapping_lengths_unique)
    #         print('if we use two sisters in the same trap vs. in different traps\n', len(sis_keys_in_overlapping_lengths),
    #               len(sis_keys_in_overlapping_lengths_unique))
    #         percentage_of_sis_trajs_in_overlapping_lengths = len(sis_keys_in_overlapping_lengths_unique) / int(len(sis_Both)/2)
    #         print('percentage_of_sis_trajs_in_overlapping_lengths\n', percentage_of_sis_trajs_in_overlapping_lengths)
    #
    #         non_sis_keys_in_overlapping_lengths = [key for key, length in zip(non_sis_array[:, 0], non_sis_array[:, 2]) if length in overlapping_lengths]
    #         print('non_sis_keys_in_overlapping_lengths\n', non_sis_keys_in_overlapping_lengths)
    #         non_sis_keys_in_overlapping_lengths_unique = np.unique([[int(s) for s in non_sis_keys_in_overlapping_lengths[ind].split('_') if s.isdigit()] for
    #                                                             ind in range(len(non_sis_keys_in_overlapping_lengths))])
    #         print('non_sis_keys_in_overlapping_lengths_unique\n', non_sis_keys_in_overlapping_lengths_unique)
    #         print('if we use two non_sisters in the same trap vs. in different traps\n', len(non_sis_keys_in_overlapping_lengths),
    #               len(non_sis_keys_in_overlapping_lengths_unique))
    #         percentage_of_non_sis_trajs_in_overlapping_lengths = len(non_sis_keys_in_overlapping_lengths_unique) / int(len(non_sis_Both)/2)
    #         print('percentage_of_non_sis_trajs_in_overlapping_lengths\n', percentage_of_non_sis_trajs_in_overlapping_lengths)
    #
    #     # Creating a distribution from which we pick generation lengths, here we get the weights
    #     weights = []
    #     for length in overlapping_lengths:
    #         sis_keys_in_overlapping_lengths = [key for key, lengthy in zip(sis_array[:, 0], sis_array[:, 2]) if abs(lengthy-length) <= 2] # lengthy == length
    #         non_sis_keys_in_overlapping_lengths = [key for key, lengthy in zip(non_sis_array[:, 0], non_sis_array[:, 2]) if abs(lengthy-length) <= 2] # lengthy == length
    #         weights.append((len(sis_keys_in_overlapping_lengths) + len(non_sis_keys_in_overlapping_lengths)) / 2)
    #
    #     # Normalize the counted weights
    #     weights = np.array(weights) / np.sum(weights)
    #
    #     # From the two datasets, pick as many pairs as the one with the lowest amount, 88 pairs in total
    #     number_of_pairs_to_pick = min(int(len(non_sis_Both)/2), int(len(sis_Both)/2))
    #
    #     # choose from said distribution of generation lengths, with replacement (better without replacement)
    #     array_of_lengths = np.random.choice(overlapping_lengths, size=number_of_pairs_to_pick, replace=True, p=weights)
    #
    #     # the trajectory dictionaries and the reference dictionaries
    #     A_dict = dict()
    #     B_dict = dict()
    #     Both_dict = dict()
    #     reference_A_dict = dict()
    #     reference_B_dict = dict()
    #
    #     # the new ordering of the sampled cells from Sisters and Nonsisters datasets
    #     new_order = 0
    #     for length in array_of_lengths:
    #         # what keys have this length for the Sisters dataset?
    #         sis_keys_in_overlapping_lengths = [key for key, lengthy in zip(sis_array[:, 0], sis_array[:, 2]) if abs(lengthy-length) <= 2] # lengthy == length
    #         # what keys have this length for the Nonsisters dataset?
    #         non_sis_keys_in_overlapping_lengths = [key for key, lengthy in zip(non_sis_array[:, 0], non_sis_array[:, 2]) if abs(lengthy-length) <= 2] # lengthy == length
    #         # choose one trajectory from the sisters and one from the nonsisters
    #         sis_choice = random.choice(sis_keys_in_overlapping_lengths)
    #         non_sis_choice = random.choice(non_sis_keys_in_overlapping_lengths)
    #         # update the returning dictionaries
    #         Both_dict.update({'A_'+str(new_order): sis_Both[str(sis_choice)[4:]],
    #                           'B_'+str(new_order): non_sis_Both[str(non_sis_choice)[8:]]})
    #         A_dict.update({str(new_order): sis_Both[str(sis_choice)[4:]]})
    #         B_dict.update({str(new_order): non_sis_Both[str(non_sis_choice)[8:]]})
    #         reference_A_dict.update({str(new_order): sis_choice})
    #         reference_B_dict.update({str(new_order): non_sis_choice})
    #         # go on to the next dataID (new_order)
    #         new_order = new_order + 1
    #
    #     return A_dict, B_dict, Both_dict, reference_A_dict, reference_B_dict

    """ Creating the Control dictionaries from the Sisters and Nonsisters dictionaries, Trace A will come from Sisters and Trace B will come from
        Nonsisters , returns A_dict, B_dict, Both_dict, reference_A_dict, reference_B_dict """

    def get_Control_A_B_and_Both_dicts(self, sis_Both, non_sis_Both, difference_criterion, **kwargs):

        def take_two_trace_not_from_the_same_trap(keys_in_overlapping_lengths, both_array, length):
            two_samples = random.choices(keys_in_overlapping_lengths, k=2)
            first = two_samples[0].split('_')
            second = two_samples[1].split('_')
            if first[0] == second[0] and first[2] == second[2]:
                print('chose two traces in the same trap')
                print(keys_in_overlapping_lengths)
                print(length)
                first_array, second_array = take_two_trace_not_from_the_same_trap(keys_in_overlapping_lengths, both_array, length)

            first_array, second_array = both_array[np.where(both_array[:, 0] == two_samples[0]), :].flatten(), both_array[np.where(both_array[:, 0] == two_samples[1]), :].flatten()

            return first_array, second_array

        # if we want some debugging data
        debug = kwargs.get('debug', [])

        # array of the key, dataframe and length of all bacteria in sis and nonsis dataset, and their concatenation
        sis_array = np.array([['sis_' + key, df_value, len(df_value)] for key, df_value in sis_Both.copy().items()])
        non_sis_array = np.array([['nonsis_' + key, df_value, len(df_value)] for key, df_value in non_sis_Both.copy().items()])

        # sis_dict = dict([('sis_' + key, df_value) for key, df_value in sis_Both.copy().items()])
        # non_sis_dict = dict([('nonsis_' + key, df_value) for key, df_value in non_sis_Both.copy().items()])
        both_array = np.concatenate((sis_array, non_sis_array), axis=0)
        # both_dict = sis_dict.copy()
        # both_dict.update(non_sis_dict)

        # print(len(both_dict), len(both_array))

        # overlapping_lengths = [ for length in np.unique([l])]
        overlapping_lengths = []
        for length in np.unique(both_array[:, 2]):
            sis_and_non_same_len = (len(np.concatenate((np.unique([[name.split('_')[ind] for ind in [0, 2]] for name in sis_array[np.where(abs(length-sis_array[:, 2]) <= difference_criterion)][:, 0]]),
                                                    np.unique([[name.split('_')[ind] for ind in [0, 2]] for name in non_sis_array[np.where(abs(length-non_sis_array[:, 2]) <= difference_criterion)][:, 0]])), axis=0)) > 2)
            if sis_and_non_same_len:
                overlapping_lengths.append(length)

        # print([length for length in overlapping_lengths if length > 100])
        # overlapping_lengths = np.array([both for both in np.unique(both_array[:, 2]) if (len(np.array(np.where(both == sis_array[:, 2])).flatten()) > 2 or
        #                                                                       len(np.array(np.where(both == non_sis_array[:, 2])).flatten()) >= 2 or
        #                                                                       len(np.array(np.where(both == both_array[:, 2])).flatten()) >= 2)])

        # to show that these "generation lengths" span 95% of all generation lengths of Sisters and Nonsisters datasets
        if debug == True:
            sis_keys_in_overlapping_lengths = [key for key, length in zip(sis_array[:, 0], sis_array[:, 2]) if length in overlapping_lengths]
            print('sis_keys_in_overlapping_lengths\n', sis_keys_in_overlapping_lengths)
            sis_keys_in_overlapping_lengths_unique = np.unique([[int(s) for s in sis_keys_in_overlapping_lengths[ind].split('_') if s.isdigit()] for
                                                                ind in range(len(sis_keys_in_overlapping_lengths))])
            print('sis_keys_in_overlapping_lengths_unique\n', sis_keys_in_overlapping_lengths_unique)
            print('if we use two sisters in the same trap vs. in different traps\n', len(sis_keys_in_overlapping_lengths),
                  len(sis_keys_in_overlapping_lengths_unique))
            percentage_of_sis_trajs_in_overlapping_lengths = len(sis_keys_in_overlapping_lengths_unique) / int(len(sis_Both) / 2)
            print('percentage_of_sis_trajs_in_overlapping_lengths\n', percentage_of_sis_trajs_in_overlapping_lengths)

            non_sis_keys_in_overlapping_lengths = [key for key, length in zip(non_sis_array[:, 0], non_sis_array[:, 2]) if
                                                   length in overlapping_lengths]
            print('non_sis_keys_in_overlapping_lengths\n', non_sis_keys_in_overlapping_lengths)
            non_sis_keys_in_overlapping_lengths_unique = np.unique(
                [[int(s) for s in non_sis_keys_in_overlapping_lengths[ind].split('_') if s.isdigit()] for
                 ind in range(len(non_sis_keys_in_overlapping_lengths))])
            print('non_sis_keys_in_overlapping_lengths_unique\n', non_sis_keys_in_overlapping_lengths_unique)
            print('if we use two non_sisters in the same trap vs. in different traps\n', len(non_sis_keys_in_overlapping_lengths),
                  len(non_sis_keys_in_overlapping_lengths_unique))
            percentage_of_non_sis_trajs_in_overlapping_lengths = len(non_sis_keys_in_overlapping_lengths_unique) / int(len(non_sis_Both) / 2)
            print('percentage_of_non_sis_trajs_in_overlapping_lengths\n', percentage_of_non_sis_trajs_in_overlapping_lengths)

        # Creating a distribution from which we pick generation lengths, here we get the weights
        weights = []
        for length in overlapping_lengths:
            sis_keys_in_overlapping_lengths = [key for key, lengthy in zip(sis_array[:, 0], sis_array[:, 2]) if lengthy == length]
            weights.append(len(sis_keys_in_overlapping_lengths))

        # Normalize the counted weights
        weights = np.array(weights) / np.sum(weights)

        # From the two datasets, pick as many pairs as the one with the lowest amount, 88 pairs in total
        number_of_pairs_to_pick = min(int(len(non_sis_Both) / 2), int(len(sis_Both) / 2))

        # choose from said distribution of generation lengths, with replacement (better without replacement)
        array_of_lengths = np.random.choice(overlapping_lengths, size=number_of_pairs_to_pick, replace=True, p=weights)

        # the trajectory dictionaries and the reference dictionaries
        A_dict = dict()
        B_dict = dict()
        Both_dict = dict()
        reference_A_dict = dict()
        reference_B_dict = dict()

        # the new ordering of the sampled cells from Sisters and Nonsisters datasets
        new_order = 0

        # so as not to repeat the same traces
        repetitions = []
        for length in array_of_lengths:
            # print(length)
            # what keys have this length for the Sisters and Nonsisters dataset?
            keys_in_overlapping_lengths = [key for key, lengthy in zip(both_array[:, 0], both_array[:, 2]) if (abs(lengthy - length) <= difference_criterion) and (key not in repetitions)]
            # print('keys_in_overlapping_lengths:', keys_in_overlapping_lengths)
            first_key = random.choice(keys_in_overlapping_lengths)
            # print(first_key)
            # print(keys_in_overlapping_lengths)
            # print([key for key in keys_in_overlapping_lengths if not (key.split('_')[0] == first_key.split('_')[0] and key.split('_')[2] == first_key.split('_')[2])])
            second_key = random.choice([key for key in keys_in_overlapping_lengths if not (key.split('_')[0] == first_key.split('_')[0] and key.split('_')[2] == first_key.split('_')[2])])
            # print(second_key)
            if first_key.split('_')[0] == second_key.split('_')[0] and first_key.split('_')[2] == second_key.split('_')[2]:
                print('chose cells in the same trap!')
                exit()

            # update the returning dictionaries
            Both_dict.update({'A_' + str(new_order): both_array[np.where(first_key == both_array[:, 0])][:, 1][0],
                              'B_' + str(new_order): both_array[np.where(second_key == both_array[:, 0])][:, 1][0]})
            A_dict.update({str(new_order): both_array[np.where(first_key == both_array[:, 0])][:, 1][0]})
            B_dict.update({str(new_order): both_array[np.where(second_key == both_array[:, 0])][:, 1][0]})
            reference_A_dict.update({str(new_order): first_key})
            reference_B_dict.update({str(new_order): second_key})
            # go on to the next dataID (new_order)
            new_order = new_order + 1
            repetitions.append(first_key)
            repetitions.append(second_key)
            
            # first_array, second_array = take_two_trace_not_from_the_same_trap(keys_in_overlapping_lengths, both_array, length)
            # 
            # # update the returning dictionaries
            # Both_dict.update({'A_' + str(new_order): first_array[1],
            #                   'B_' + str(new_order): second_array[1]})
            # A_dict.update({str(new_order): first_array[1]})
            # B_dict.update({str(new_order): second_array[1]})
            # reference_A_dict.update({str(new_order): first_array[0]})
            # reference_B_dict.update({str(new_order): second_array[0]})
            # # go on to the next dataID (new_order)
            # new_order = new_order + 1
            # repetitions.append(first_array[0])
            # repetitions.append(second_array[0])

        return A_dict, B_dict, Both_dict, reference_A_dict, reference_B_dict

    """ Plot the A/B trajectories that make up the A/B traces of Control, doesn't return anything """
    def check_raw_data_of_Control(self):
        # load first sheet of each Excel-File, fill internal data structure
        for A_dataID, B_dataID in zip(self.reference_A_dict.values(), self.reference_B_dict.values()):

            # see if the A trajectory comes from the original A trace and same for the B trajectory
            if A_dataID[4] == 'A':
                A_keylist = self._keylist[:5]
            if A_dataID[4] == 'B':
                A_keylist = self._keylist[5:]
            if B_dataID[8] == 'A':
                B_keylist = self._keylist[:5]
            if B_dataID[8] == 'B':
                B_keylist = self._keylist[5:]

            # convert from string to integers
            old_A_dataID = A_dataID
            old_B_dataID = B_dataID
            A_dataID = int(A_dataID[6:])
            B_dataID = int(B_dataID[10:])

            # Here we check if the traces should be together or not
            plt.plot(self.sister_data[A_dataID][A_keylist[0]], self.sister_data[A_dataID][A_keylist[1]], color='blue', label=old_A_dataID)
            plt.plot(self.nonsister_data[B_dataID][B_keylist[0]], self.nonsister_data[B_dataID][B_keylist[1]], color='orange', label=old_B_dataID)
            plt.show()
            plt.close()

    def __init__(self, **kwargs):

        # Initialize the Sister class to get Sister data
        Sister.__init__(self, **kwargs)
        self.sister_data = self._data.copy()
        self.sister_dataorigin = self._origin.copy()
        self.sis_Both = self.Both_dict.copy()
        self.sis_all_protein = self._dataset_raw_protein_data_dict.copy()

        # Initialize the Nonsister class to get Nonsister data
        Nonsister.__init__(self, **kwargs)
        self.nonsister_data = self._data.copy()
        self.nonsister_dataorigin = self._origin.copy()
        self.non_sis_Both = self.Both_dict.copy()
        self._keylist = self._keylist[:-1].copy() # this is because one of the excel files has an error and uses fluoresenceb instead of fluoresenceB,
        # we will adjust to this
        self.nonsis_all_protein = self._dataset_raw_protein_data_dict.copy()

        # Get the dictionaries
        self.A_dict, self.B_dict, self.Both_dict, self.reference_A_dict, self.reference_B_dict = self.get_Control_A_B_and_Both_dicts(
            sis_Both=self.sis_Both, non_sis_Both=self.non_sis_Both, difference_criterion=3) # debug=True is an option
        print('Got the dictionaries and the reference dictionaries for Control Subclass')

        # Get the dictionaries for protein
        self.A_raw_protein_data_dict, self.B_raw_protein_data_dict, self._dataset_raw_protein_data_dict, \
        self.reference_A_raw_protein_data_dict, self.reference_B_raw_protein_data_dict = self.get_Control_A_B_and_Both_dicts(
            sis_Both=self.sis_all_protein, non_sis_Both=self.nonsis_all_protein, difference_criterion=50)  # debug=True is an option
        print('Got the protein dictionaries and the reference dictionaries for Control Subclass')

        # lists to store data internally
        self._data = list()
        self.A_dataorigin = list()
        self.B_dataorigin = list()

        # load first sheet of each Excel-File, fill internal data structure
        for A_data, B_data in zip(self.reference_A_dict.values(), self.reference_B_dict.values()):

            A_dataID = A_data.split('_')
            B_dataID = B_data.split('_')

            # see if the A trajectory comes from the original A trace and same for the B trajectory
            if A_dataID[1] == 'A':
                A_keylist = self._keylist[:5]
            if A_dataID[1] == 'B':
                A_keylist = self._keylist[5:]
            if B_dataID[1] == 'A':
                B_keylist = self._keylist[:5]
            if B_dataID[1] == 'B':
                B_keylist = self._keylist[5:]

            # print(A_dataID, B_dataID)

            # convert from string to integers
            A_dataID = int(A_dataID[2])
            B_dataID = int(B_dataID[2])

            if A_data.split('_')[0] == 'sis':
                A_dat = self.sister_data
                A_origin = self.sister_dataorigin
            else:
                A_dat = self.nonsister_data
                A_origin = self.nonsister_dataorigin

            if B_data.split('_')[0] == 'sis':
                B_dat = self.sister_data
                B_origin = self.sister_dataorigin
            else:
                B_dat = self.nonsister_data
                B_origin = self.nonsister_dataorigin

            # check if this is the instance in which there is a grammar mistake in the excel file...
            if 'fluorescenceB' not in list(A_dat[A_dataID].keys()):
                A_keylist = ['fluorescenceb' if x == 'fluorescenceB' else x for x in A_keylist]
                print('WE CHOOSE A TRACE WITH THE EXCEL FILE HAVING THE FLUORESENCE MISTAKE, ' + str(A_dataID))
            if 'fluorescenceB' not in list(B_dat[B_dataID].keys()):
                B_keylist = ['fluorescenceb' if x == 'fluorescenceB' else x for x in B_keylist]
                print('WE CHOOSE A TRACE WITH THE EXCEL FILE HAVING THE FLUORESENCE MISTAKE, ' + str(B_dataID))

            # the _data contains all dataframes from the excel files in the directory __infiles
            self._data.append(pd.concat([A_dat[A_dataID][A_keylist], B_dat[B_dataID][B_keylist]]))
            # the name of the excel file
            self.A_dataorigin.append(A_origin[A_dataID])
            self.B_dataorigin.append(B_origin[B_dataID])

        # use this for the loops later on
        self._data_len = len(self._data)

        # there's no point in not having data ...
        # ... or something went wrong. rather stop here
        if not self._data_len > 0:
            raise IOError('no data loaded')

        self.A_stats_dict, self.B_stats_dict, self.trap_stats_dict = self.trap_and_traj_variable_statistics(A_dict=self.A_dict, B_dict=self.B_dict)
        print('Got the A_stats_dict, B_stats_dict, and trap_stats_dict attributes for the Control class')

        self.Trap_coeffs_of_vars_dict = dict()
        # Get the coefficients of variation
        for key, val in zip(self.trap_stats_dict.keys(), self.trap_stats_dict.values()):
            Trap_coeffs_of_vars = self.coeffs_of_variations(stats_df=val)
            self.Trap_coeffs_of_vars_dict.update({key: Trap_coeffs_of_vars})

        self.A_coeffs_of_vars_dict = dict()
        # Get the coefficients of variation
        for key, val in zip(self.A_stats_dict.keys(), self.A_stats_dict.values()):
            A_coeffs_of_vars = self.coeffs_of_variations(stats_df=val)
            self.A_coeffs_of_vars_dict.update({key: A_coeffs_of_vars})

        self.B_coeffs_of_vars_dict = dict()
        # Get the coefficients of variation
        for key, val in zip(self.B_stats_dict.keys(), self.B_stats_dict.values()):
            B_coeffs_of_vars = self.coeffs_of_variations(stats_df=val)
            self.B_coeffs_of_vars_dict.update({key: B_coeffs_of_vars})
        print('Got the coefficients of variation for A/B traces and traps for the Sister class')

        # log some variables in the gen_dicts
        self._log_A_dict = self.log_the_gen_dict(dictionary=self.A_dict, variables_to_log=['length_birth', 'length_final',
                                                                                           'division_ratio'])
        self._log_B_dict = self.log_the_gen_dict(dictionary=self.B_dict, variables_to_log=['length_birth', 'length_final',
                                                                                           'division_ratio'])
        self._log_Both_dict = self.log_the_gen_dict(dictionary=self.Both_dict, variables_to_log=['length_birth', 'length_final',
                                                                                                 'division_ratio'])
        print('Got the _log_A/B/Both_dict attributes for Control')

        # Get the relationship dataframes, made with the normalized lengths
        self.A_intra_gen_bacteria, self.B_intra_gen_bacteria = self.intragenerational_dataframe_creations(
            A_dict=self.A_dict, B_dict=self.B_dict, how_many_cousins_and_sisters=6)
        print('got the intragenerational_dataframes for Sister')

        # contains all the attributes and methods
        self.organizational_table = self.table_with_attributes_and_available_methods()
        print('got the organizational table for Control')


def output_the_attributes_and_methods_of_classes(Population, Sister, Nonsister, Control, path):
    """ here we output to a csv all the attributes and methods for certain classes for reference """
    c_dict = Control.table_with_attributes_and_available_methods()
    s_dict = Sister.table_with_attributes_and_available_methods()
    n_dict = Nonsister.table_with_attributes_and_available_methods()
    p_dict = Population.table_with_attributes_and_available_methods()

    attributes_table = pd.DataFrame(columns=['Class', 'Attribute'])
    for att in list(set(c_dict['Attributes'] + s_dict['Attributes'] + n_dict['Attributes'] + p_dict['Attributes'])):
        class_name = ''
        if att in p_dict['Attributes']:
            if class_name == '':
                class_name = 'Population'
            else:
                class_name = class_name + ', ' + 'Population'
        if att in s_dict['Attributes']:
            if class_name == '':
                class_name = 'Sister'
            else:
                class_name = class_name + ', ' + 'Sister'
        if att in n_dict['Attributes']:
            if class_name == '':
                class_name = 'Nonsister'
            else:
                class_name = class_name + ', ' + 'Nonsister'
        if att in c_dict['Attributes']:
            if class_name == '':
                class_name = 'Control'
            else:
                class_name = class_name + ', ' + 'Control'
        if class_name == 'Population, Sister, Nonsister, Control':
            class_name = 'All'

        attributes_table = attributes_table.append({'Class': class_name, 'Attribute': att}, ignore_index=True)

    methods_table = pd.DataFrame(columns=['Class', 'Methods'])
    for method in list(set(c_dict['Methods'] + s_dict['Methods'] + n_dict['Methods'] + p_dict['Methods'])):
        class_name = ''
        if method in p_dict['Methods']:
            if class_name == '':
                class_name = 'Population'
            else:
                class_name = class_name + ', ' + 'Population'
        if method in s_dict['Methods']:
            if class_name == '':
                class_name = 'Sister'
            else:
                class_name = class_name + ', ' + 'Sister'
        if method in n_dict['Methods']:
            if class_name == '':
                class_name = 'Nonsister'
            else:
                class_name = class_name + ', ' + 'Nonsister'
        if method in c_dict['Methods']:
            if class_name == '':
                class_name = 'Control'
            else:
                class_name = class_name + ', ' + 'Control'
        if class_name == 'Population, Sister, Nonsister, Control':
            class_name = 'All'

        methods_table = methods_table.append({'Class': class_name, 'Methods': method}, ignore_index=True)

    # output it
    attributes_table.to_csv(path_or_buf=path+'NewSisterCellClass_attributes.csv', index=False)
    methods_table.to_csv(path_or_buf=path+'NewSisterCellClass_methods.csv', index=False)


def save_to_pickle_and_output_tables():
    # For Mac
    infiles_sisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/SISTERS/*.xls')
    infiles_nonsisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/NONSISTERS/*.xls')
    infiles_sisters_protein = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/ALLSIS/*.xls')
    infiles_nonsisters_protein = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/AllNONSIS/*.xls')

    print('GETTING POPULATION')

    Population = ssc.Population(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length',
                                infiles_sister_protein=infiles_sisters_protein, infiles_nonsister_protein=infiles_nonsisters_protein)

    print('GETTING SISTER')

    Sister = ssc.Sister(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length',
                        infiles_sister_protein=infiles_sisters_protein, infiles_nonsister_protein=infiles_nonsisters_protein, bare_minimum_pop=True)

    print('GETTING NONSISTER')

    Nonsister = ssc.Nonsister(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length',
                              infiles_sister_protein=infiles_sisters_protein, infiles_nonsister_protein=infiles_nonsisters_protein,
                              bare_minimum_pop=True)

    print('GETTING CONTROL')

    Control = ssc.Control(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length',
                          bare_minimum_pop=True, bare_minimum_sis=True, bare_minimum_non_sis=True,
                          infiles_sister_protein=infiles_sisters_protein, infiles_nonsister_protein=infiles_nonsisters_protein)

    print('GETTING ENVIRONMENTAL SISTERS')

    Env_Sister = ssc.Env_Sister(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length',
                                infiles_sister_protein=infiles_sisters_protein, infiles_nonsister_protein=infiles_nonsisters_protein,
                                bare_minimum_pop=True)

    print('GETTING ENVIRONMENTAL NONSISTERS')

    Env_Nonsister = ssc.Env_Nonsister(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length',
                                infiles_sister_protein=infiles_sisters_protein, infiles_nonsister_protein=infiles_nonsisters_protein,
                                bare_minimum_pop=True)

    print('OUTPUTTING THE ATTRIBUTES AND METHODS FOR THE CLASSES TO CSV')
    output_the_attributes_and_methods_of_classes(Population, Sister, Nonsister, Control, r'/Users/alestawsky/PycharmProjects/untitled/')

    # Create many pickles and dump the respective classes into them
    pickle_out = open("NewSisterCellClass_Population.pickle", "wb")
    pickle.dump(Population, pickle_out)
    pickle_out.close()

    pickle_out = open("NewSisterCellClass_Sister.pickle", "wb")
    pickle.dump(Sister, pickle_out)
    pickle_out.close()

    pickle_out = open("NewSisterCellClass_Nonsister.pickle", "wb")
    pickle.dump(Nonsister, pickle_out)
    pickle_out.close()

    pickle_out = open("NewSisterCellClass_Control.pickle", "wb")
    pickle.dump(Control, pickle_out)
    pickle_out.close()

    pickle_out = open("NewSisterCellClass_Env_Sister.pickle", "wb")
    pickle.dump(Env_Sister, pickle_out)
    pickle_out.close()

    pickle_out = open("NewSisterCellClass_Env_Nonsister.pickle", "wb")
    pickle.dump(Env_Nonsister, pickle_out)
    pickle_out.close()

    print('SAVED TO PICKLE')


def debugging_by_importing_the_data():
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

    """ the linear regression """
    # mother_df = Population.mother_dfs[0].copy()
    # daughter_df = Population.daughter_dfs[0].copy()
    # # make the fold growth equal to delta phi
    # mother_df['fold_growth'] = mother_df['fold_growth'] - Population.pop_stats['fold_growth'].loc['mean']
    # daughter_df['fold_growth'] = daughter_df['fold_growth'] - Population.pop_stats['fold_growth'].loc['mean']
    # factor_vars = ['length_birth', 'fold_growth']
    #
    # mat, scores, intercepts = Population.linear_regression_framework(df_of_avgs=mother_df[['traj_avg_' + fv for fv in factor_vars]],
    #                                                                  factor_variables=factor_vars,
    #                                                                  target_variables=['length_birth', 'fold_growth'], factor_df=mother_df,
    #                                                                  target_df=daughter_df, fit_intercept=False)
    #
    # print('regression matrix\n', mat)
    # print('scores\n', scores)
    # print('intercepts\n', intercepts)

    """ plot the normed fluorescence """
    # print(Nonsister.A_raw_protein_data_dict.keys())
    # plt.plot(Nonsister.A_raw_protein_data_dict['0']['normed_fluor'])
    # plt.plot(Nonsister.B_raw_protein_data_dict['0']['normed_fluor'])
    # plt.show()
    # exit()

    """ plot the cross correlations """
    # for A_prot_key, B_prot_key in zip(Nonsister.A_raw_protein_data_dict.keys(), Nonsister.B_raw_protein_data_dict.keys()):
    #     Nonsister.cross_correlation(vec_1=Nonsister.A_raw_protein_data_dict[A_prot_key]['normed_fluor'], label_1='non sis A '+str(A_prot_key),
    #                              vec_2=Nonsister.B_raw_protein_data_dict[B_prot_key]['normed_fluor'], label_2='non sis B '+str(B_prot_key), plot=True,
    #                              title='Nonsister dataset')

    """ Plotting the histograms of the coefficient of covariance """
    # Sister.plot_hist_of_coefs_of_variance()
    #
    # Nonsister.plot_hist_of_coefs_of_variance()
    #
    # Control.plot_hist_of_coefs_of_variance()


if __name__ == '__main__':
    # For Mac
    infiles_sisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/SISTERS/*.xls')
    infiles_nonsisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/NONSISTERS/*.xls')
    infiles_sisters_protein = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/ALLSIS/*.xls')
    infiles_nonsisters_protein = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/AllNONSIS/*.xls')

    Global_Mean = ssc.Global_Mean(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', bare_minimum_pop=True, bare_minimum_sis=True,
        bare_minimum_non_sis=True, infiles_sister_protein=infiles_sisters_protein, infiles_nonsister_protein=infiles_nonsisters_protein, what_to_subtract='global mean', start_index=None, end_index=None)

    print(Global_Mean)
    print(Global_Mean.__dir__())
    print(Global_Mean.Sister.__dir__())
    print(Global_Mean.Sister.all_bacteria)
    print(len(Global_Mean.Sister.A_dict), Global_Mean.Sister.A_dict)

    exit()

    # creates the pickle and saves it to directory we are automatically working under
    save_to_pickle_and_output_tables()
    # debugging_by_importing_the_data()
