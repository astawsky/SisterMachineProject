import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import os
import scipy.stats as stats
import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
import seaborn as sns
import iteround

""" For making the Seaborn Plots look better """


def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(r"$\rho = {:.2f}$".format(r), xy=(.1, .9), xycoords=ax.transAxes, fontweight='bold', fontsize=16)


""" For making the Seaborn Plots look better """


def limit_the_axes(x, y, **kws):
    ax = plt.gca()
    ax.set_xlim([-3 * np.std(x) + np.mean(x), 5 * np.std(x) + np.mean(x)])
    ax.set_ylim([-3 * np.std(y) + np.mean(y), 5 * np.std(y) + np.mean(y)])


def plot_the_label(var1, **kwargs):
    ax = plt.gca()
    ax.annotate(r'${:.3}\pm{:.3}$'.format(np.mean(var1), np.std(var1)), xy=(.1, .9), xycoords=ax.transAxes, fontweight='bold')


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
    mom_variable_symbol = [r'$\tau_n$', r'$x_n(0)$', r'$x_n(\tau)$', r'$\alpha_n$', r'$\phi_n$', r'$f_n$', r'$\Delta_n$']
    daughter_variable_symbol = [r'$\tau_{n+1}$', r'$x_{n+1}(0)$', r'$x_{n+1}(\tau)$', r'$\alpha_{n+1}$', r'$\phi_{n+1}$', r'$f_{n+1}$', r'$\Delta_{n+1}$']
    same_cell_variable_symbol = [r'$\tau$', r'$x(0)$', r'$x(\tau)$', r'$\alpha$', r'$\phi$', r'$f$', r'$\Delta$']
    A_variable_symbols = dict(zip(_variable_names, [r'\ln(\tau)_A', r'\ln(x(0))_A', r'\ln(x(\tau))_A', r'\ln(\alpha)_A', r'\ln(\phi)_A', r'f_A', r'\Delta_A']))
    B_variable_symbols = dict(zip(_variable_names, [r'\ln(\tau)_B', r'\ln(x(0))_B', r'\ln(x(\tau))_B', r'\ln(\alpha)_B', r'\ln(\phi)_B', r'f_B', r'\Delta_B']))

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

    """ BEGINNING: Main Definitions """



    # A function that goes into the init function and gets the corresponding protein raw data based on the list what_class

    def get_protein_raw_data(self, **kwargs):
        sister_protein_data = list()
        sister_protein_origin = list()
        sister_protein_keylist = list()
        nonsister_protein_data = list()
        nonsister_protein_origin = list()
        nonsister_protein_keylist = list()

        datasets = kwargs.get('datasets', [])

        # where the excel files are
        sister_protein_files = kwargs.get('infiles_sister_protein', [])

        # load first sheet of each Excel-File, fill internal data structure
        for filename in sister_protein_files:
            try:
                # creates a dataframe from the excel file
                tmpdata = pd.read_excel(filename)
            except:
                continue
            # the _data contains all dataframes from the excel files in the directory _infiles
            sister_protein_data.append(tmpdata)
            # the name of the excel file
            sister_protein_origin.append(filename)
            for k in tmpdata.keys():
                if not str(k) in sister_protein_keylist:
                    # this list contains the column names of the excel file
                    sister_protein_keylist.append(str(k))

        nonsister_protein_files = kwargs.get('infiles_nonsister_protein', [])
        # load first sheet of each Excel-File, fill internal data structure
        for filename in nonsister_protein_files:
            try:
                # creates a dataframe from the excel file
                tmpdata = pd.read_excel(filename)
            except:
                continue
            # the _data contains all dataframes from the excel files in the directory _infiles
            nonsister_protein_data.append(tmpdata)
            # the name of the excel file
            nonsister_protein_origin.append(filename)
            for k in tmpdata.keys():
                if not str(k) in nonsister_protein_keylist:
                    # this list contains the column names of the excel file
                    nonsister_protein_keylist.append(str(k))

        # there's no point in not having data ...
        # ... or something went wrong. rather stop here
        if (len(sister_protein_data) <= 0) or (len(nonsister_protein_data) <= 0):
            raise IOError('no data loaded')

        if 'Population' in datasets:
            self.pop_all_raw_protein_data_dict, self.pop_A_raw_protein_data_dict, self.pop_B_raw_protein_data_dict = self.create_dictionaries_of_mean_fluor(
                protein_data_len=len(sister_protein_data + nonsister_protein_data), protein_data=sister_protein_data + nonsister_protein_data)
            print('Got the _all_raw_protein_data_dict and A/B_raw_protein_data_dict attribute in Population')
        if 'Sisters' in datasets:
            self.sis_all_raw_protein_data_dict, self.sis_A_raw_protein_data_dict, self.sis_B_raw_protein_data_dict = self.create_dictionaries_of_mean_fluor(protein_data_len=len(sister_protein_data),
                protein_data=sister_protein_data)
            print('Got the _all_raw_protein_data_dict and A/B_raw_protein_data_dict attribute in Sister')
        if 'Nonsisters' in datasets:
            self.non_sis_all_raw_protein_data_dict, self.non_sis_A_raw_protein_data_dict, self.non_sis_B_raw_protein_data_dict = self.create_dictionaries_of_mean_fluor(
                protein_data_len=len(nonsister_protein_data), protein_data=nonsister_protein_data)
            print('Got the _all_raw_protein_data_dict and A/B_raw_protein_data_dict attribute in Nonsister')
        if 'Control' in datasets:
            if 'Sisters' not in datasets:
                self.sis_all_raw_protein_data_dict, self.sis_A_raw_protein_data_dict, self.sis_B_raw_protein_data_dict = self.create_dictionaries_of_mean_fluor(
                    protein_data_len=len(sister_protein_data), protein_data=sister_protein_data)
                print('Got the _all_raw_protein_data_dict and A/B_raw_protein_data_dict attribute in Sister')
            if 'Nonsisters' not in datasets:
                self.non_sis_all_raw_protein_data_dict, self.non_sis_A_raw_protein_data_dict, self.non_sis_B_raw_protein_data_dict = self.create_dictionaries_of_mean_fluor(
                    protein_data_len=len(nonsister_protein_data), protein_data=nonsister_protein_data)
                print('Got the _all_raw_protein_data_dict and A/B_raw_protein_data_dict attribute in Nonsister')
            # Get the dictionaries for protein
            self.con_A_raw_protein_data_dict, self.con_B_raw_protein_data_dict, self.con_Both_raw_protein_data_dict, self.con_log_A_raw_protein_data_dict, self.con_log_B_raw_protein_data_dict, self.con_log_Both_raw_protein_data_dict, self.reference_A_raw_protein_data_dict, self.reference_B_raw_protein_data_dict = self.get_Control_A_B_and_Both_dicts(
                sis_Both=self.sis_all_raw_protein_data_dict, non_sis_Both=self.non_sis_all_raw_protein_data_dict, difference_criterion=50)  # debug=True is an option
            print('Got the _all_raw_protein_data_dict and A/B_raw_protein_data_dict attribute in Control')

    # A function that goes into the init function and gets the corresponding protein raw data based on the list what_class

    def gen_dicts_and_class_stats(self, **kwargs):

        # The observable we will use to see when the cell divided and what the cycle data is based on (can be either 'length',
        # 'cellarea', or 'fluorescence')
        _discretization_variable = kwargs.get('discretization_variable', [])
        self.what_to_subtract = kwargs.get('what_to_subtract', None)
        start_index = kwargs.get('start_index', None)
        end_index = kwargs.get('end_index', None)
        self.datasets = kwargs.get('datasets', [])

        # where to put the raw data
        sister_data = list()
        sister_origin = list()
        sister_keylist = list()
        nonsister_data = list()
        nonsister_origin = list()
        nonsister_keylist = list()

        # where the excel files are
        sister_files = kwargs.get('infiles_sister', [])

        # load first sheet of each Excel-File, fill internal data structure
        for filename in sister_files:
            try:
                # creates a dataframe from the excel file
                tmpdata = pd.read_excel(filename)
            except:
                continue
            # the _data contains all dataframes from the excel files in the directory _infiles
            sister_data.append(tmpdata)
            # the name of the excel file
            sister_origin.append(filename)
            for k in tmpdata.keys():
                if not str(k) in sister_keylist:
                    # this list contains the column names of the excel file
                    sister_keylist.append(str(k))

        nonsister_files = kwargs.get('infiles_nonsister', [])
        # load first sheet of each Excel-File, fill internal data structure
        for filename in nonsister_files:
            try:
                # creates a dataframe from the excel file
                tmpdata = pd.read_excel(filename)
            except:
                continue
            # the _data contains all dataframes from the excel files in the directory _infiles
            nonsister_data.append(tmpdata)
            # the name of the excel file
            nonsister_origin.append(filename)
            for k in tmpdata.keys():
                if not str(k) in nonsister_keylist:
                    # this list contains the column names of the excel file
                    nonsister_keylist.append(str(k))

        # there's no point in not having data ...
        # ... or something went wrong. rather stop here
        if (len(sister_data) <= 0) or (len(nonsister_data) <= 0):
            raise IOError('no data loaded')

        # If we want to subtract the global statistic we have to get the all bacteria dataset first and then go through the process
        if self.what_to_subtract is not None:
            if self.what_to_subtract.split(' ')[0] == 'global':
                print('global statistic, so will calculate the all bacteria df and then the rest')
                self.pop_Both, self.pop_A, self.pop_B, self.pop_log_Both, self.pop_log_A, self.pop_log_B = self.create_dictionaries_of_traces(data_len=len(sister_data + nonsister_data),
                    discretization_variable=_discretization_variable, _keylist=sister_keylist + nonsister_keylist, _data=sister_data + nonsister_data, what_to_subtract=self.what_to_subtract,
                    start_index=start_index, end_index=end_index)

                # Get the all bacteria dataset
                global_df = self.create_all_bacteria_dict(self.pop_Both).drop(columns='trap_ID')
                log_global_df = self.create_all_bacteria_dict(self.pop_log_Both).drop(columns='trap_ID')
            else:
                global_df, log_global_df = None, None
        else:
            global_df, log_global_df = None, None

        if 'Population' in self.datasets:
            self.pop_Both, self.pop_A, self.pop_B, self.pop_log_Both, self.pop_log_A, self.pop_log_B, self.pop_inside_generation_A_dict, self.pop_inside_generation_B_dict = self.create_dictionaries_of_traces(
                data_len=len(sister_data + nonsister_data), discretization_variable=_discretization_variable, _keylist=sister_keylist + nonsister_keylist, _data=sister_data + nonsister_data,
                what_to_subtract=self.what_to_subtract, start_index=start_index, end_index=end_index, global_df=global_df, log_global_df=log_global_df)
            print('Population Generation data Finished')

            # pool all the dictionaries into a big dataframe
            self.pop_A_pooled, self.pop_B_pooled = self.get_pooled_trap_dict(self.pop_A, self.pop_B, self._variable_names)
            print('Got the pooled dataframe for Population')

            # pool all the dictionaries into a big dataframe
            self.pop_log_A_pooled, self.pop_log_B_pooled = self.get_pooled_trap_dict(self.pop_log_A, self.pop_log_B, self._variable_names)
            print('Got the log pooled dataframe for Population')

            # Get the all bacteria dataset
            self.all_bacteria = self.create_all_bacteria_dict(self.pop_Both)
            self.log_all_bacteria = self.create_all_bacteria_dict(self.pop_log_Both)
            print('Got the all bacteria dataframe')

            # Get the mother daughter arrays of the dfs
            self.mother_dfs, self.daughter_dfs = self.intergenerational_dataframe_creations(all_data_dict=self.pop_Both, how_many_separations=16)
            print('Got the mother array and daughter array dataframes in Population')

            # Get the log mother daughter arrays of the dfs
            self.log_mother_dfs, self.log_daughter_dfs = self.intergenerational_dataframe_creations(all_data_dict=self.pop_log_Both, how_many_separations=16)
            print('Got the (log) mother array and daughter array dataframes in Population')
        if 'Sisters' in self.datasets:
            self.sis_Both, self.sis_A, self.sis_B, self.sis_log_Both, self.sis_log_A, self.sis_log_B, self.sis_inside_generation_A_dict, self.sis_inside_generation_B_dict = self.create_dictionaries_of_traces(
                data_len=len(sister_data), discretization_variable=_discretization_variable, _keylist=sister_keylist, _data=sister_data, what_to_subtract=self.what_to_subtract,
                start_index=start_index, end_index=end_index, global_df=global_df, log_global_df=log_global_df)
            print('Sister Generation data Finished')

            # pool all the dictionaries into a big dataframe
            self.sis_A_pooled, self.sis_B_pooled = self.get_pooled_trap_dict(self.sis_A, self.sis_B, self._variable_names)
            print('Got the pooled dataframe for Sisters')

            # pool all the dictionaries into a big dataframe
            self.sis_log_A_pooled, self.sis_log_B_pooled = self.get_pooled_trap_dict(self.sis_log_A, self.sis_log_B, self._variable_names)
            print('Got the log pooled dataframe for Sisters')

            # Get the relationship dataframes, made with the normalized lengths
            self.sis_A_intra_gen_bacteria, self.sis_B_intra_gen_bacteria = self.intragenerational_dataframe_creations(A_dict=self.sis_A, B_dict=self.sis_B, how_many_cousins_and_sisters=6)
            print('got the intragenerational_dataframes for Sister')

            # Get the relationship dataframes, made with the normalized lengths
            self.sis_log_A_intra_gen_bacteria, self.sis_log_B_intra_gen_bacteria = self.intragenerational_dataframe_creations(A_dict=self.sis_log_A, B_dict=self.sis_log_B,
                how_many_cousins_and_sisters=6)
            print('got the (log) intragenerational_dataframes for Sister')
        if 'Nonsisters' in self.datasets:
            self.non_sis_Both, self.non_sis_A, self.non_sis_B, self.non_sis_log_Both, self.non_sis_log_A, self.non_sis_log_B, self.non_sis_inside_generation_A_dict, self.non_sis_inside_generation_B_dict = self.create_dictionaries_of_traces(
                data_len=len(nonsister_data), discretization_variable=_discretization_variable, _keylist=nonsister_keylist, _data=nonsister_data, what_to_subtract=self.what_to_subtract,
                start_index=start_index, end_index=end_index, global_df=global_df, log_global_df=log_global_df)
            print('Non-Sister Generation data Finished')

            # pool all the dictionaries into a big dataframe
            self.non_sis_A_pooled, self.non_sis_B_pooled = self.get_pooled_trap_dict(self.non_sis_A, self.non_sis_B, self._variable_names)
            print('Got the pooled dataframe for Non-sisters')

            # pool all the dictionaries into a big dataframe
            self.non_sis_log_A_pooled, self.non_sis_log_B_pooled = self.get_pooled_trap_dict(self.non_sis_log_A, self.non_sis_log_B, self._variable_names)
            print('Got the log pooled dataframe for Non-Sisters')

            # Get the relationship dataframes, made with the normalized lengths
            self.non_sis_A_intra_gen_bacteria, self.non_sis_B_intra_gen_bacteria = self.intragenerational_dataframe_creations(A_dict=self.non_sis_A, B_dict=self.non_sis_B,
                how_many_cousins_and_sisters=6)
            print('got the intragenerational_dataframes for Non-Sisters')

            # Get the relationship dataframes, made with the normalized lengths
            self.non_sis_log_A_intra_gen_bacteria, self.non_sis_log_B_intra_gen_bacteria = self.intragenerational_dataframe_creations(A_dict=self.non_sis_log_A, B_dict=self.non_sis_log_B,
                how_many_cousins_and_sisters=6)
            print('got the (log) intragenerational_dataframes for Non-Sisters')
        if 'Control' in self.datasets:
            if 'Sisters' not in self.datasets:
                self.sis_Both, self.sis_A, self.sis_B, self.sis_log_Both, self.sis_log_A, self.sis_log_B, self.sis_inside_generation_A_dict, self.sis_inside_generation_B_dict = self.create_dictionaries_of_traces(
                    data_len=len(sister_data), discretization_variable=_discretization_variable, _keylist=sister_keylist, _data=sister_data, what_to_subtract=self.what_to_subtract,
                    start_index=start_index, end_index=end_index, global_df=global_df, log_global_df=log_global_df)
                print('Sister Generation data Finished')

            if 'Nonsisters' not in self.datasets:
                self.non_sis_Both, self.non_sis_A, self.non_sis_B, self.non_sis_log_Both, self.non_sis_log_A, self.non_sis_log_B, self.non_sis_inside_generation_A_dict, self.non_sis_inside_generation_B_dict = self.create_dictionaries_of_traces(
                    data_len=len(nonsister_data), discretization_variable=_discretization_variable, _keylist=nonsister_keylist, _data=nonsister_data, what_to_subtract=self.what_to_subtract,
                    start_index=start_index, end_index=end_index, global_df=global_df, log_global_df=log_global_df)
                print('Non-Sister Generation data Finished')

            """ Now we do the Control dataset """

            self.con_A, self.con_B, self.con_Both, self.con_log_A, self.con_log_B, self.con_log_Both, self.reference_A_dict, self.reference_B_dict = self.get_Control_A_B_and_Both_dicts(
                sis_Both=self.sis_Both, non_sis_Both=self.non_sis_Both, sis_log_Both=self.sis_log_Both, non_sis_log_Both=self.non_sis_log_Both, difference_criterion=3)  # debug=True is an option
            print('Control Generation data Finished')

            # pool all the dictionaries into a big dataframe
            self.con_A_pooled, self.con_B_pooled = self.get_pooled_trap_dict(self.con_A, self.con_B, self._variable_names)
            print('Got the pooled dataframe for Control')

            # pool all the dictionaries into a big dataframe
            self.con_log_A_pooled, self.con_log_B_pooled = self.get_pooled_trap_dict(self.con_log_A, self.con_log_B, self._variable_names)
            print('Got the log pooled dataframe for Control')

            # Get the relationship dataframes, made with the normalized lengths
            self.con_A_intra_gen_bacteria, self.con_B_intra_gen_bacteria = self.intragenerational_dataframe_creations(A_dict=self.con_A, B_dict=self.con_B, how_many_cousins_and_sisters=6)
            print('got the intragenerational_dataframes for Control')

            # Get the relationship dataframes, made with the normalized lengths
            self.con_log_A_intra_gen_bacteria, self.con_log_B_intra_gen_bacteria = self.intragenerational_dataframe_creations(A_dict=self.con_log_A, B_dict=self.con_log_B,
                how_many_cousins_and_sisters=6)
            print('got the (log) intragenerational_dataframes for Control')

            # lists to store data internally
            control_data = list()
            control_A_data_origin = list()
            control_B_data_origin = list()
            keylist = nonsister_keylist[:-1].copy()

            # load first sheet of each Excel-File, fill internal data structure
            for A_data, B_data in zip(self.reference_A_dict.values(), self.reference_B_dict.values()):

                A_dataID = A_data.split('_')
                B_dataID = B_data.split('_')

                # see if the A trajectory comes from the original A trace and same for the B trajectory
                if A_dataID[1] == 'A':
                    A_keylist = keylist[:5]
                if A_dataID[1] == 'B':
                    A_keylist = keylist[5:]
                if B_dataID[1] == 'A':
                    B_keylist = keylist[:5]
                if B_dataID[1] == 'B':
                    B_keylist = keylist[5:]

                # convert from string to integers
                A_dataID = int(A_dataID[2])
                B_dataID = int(B_dataID[2])

                if A_data.split('_')[0] == 'sis':
                    A_dat = sister_data
                    A_origin = sister_origin
                else:
                    A_dat = nonsister_data
                    A_origin = nonsister_origin

                if B_data.split('_')[0] == 'sis':
                    B_dat = sister_data
                    B_origin = sister_origin
                else:
                    B_dat = nonsister_data
                    B_origin = nonsister_origin

                # check if this is the instance in which there is a grammar mistake in the excel file...
                if 'fluorescenceB' not in list(A_dat[A_dataID].keys()):
                    A_keylist = ['fluorescenceb' if x == 'fluorescenceB' else x for x in A_keylist]
                    print('WE CHOOSE A TRACE WITH THE EXCEL FILE HAVING THE FLUORESENCE MISTAKE, ' + str(A_dataID))
                if 'fluorescenceB' not in list(B_dat[B_dataID].keys()):
                    B_keylist = ['fluorescenceb' if x == 'fluorescenceB' else x for x in B_keylist]
                    print('WE CHOOSE A TRACE WITH THE EXCEL FILE HAVING THE FLUORESENCE MISTAKE, ' + str(B_dataID))

                # the _data contains all dataframes from the excel files in the directory __infiles
                control_data.append(pd.concat([A_dat[A_dataID][A_keylist], B_dat[B_dataID][B_keylist]]))
                # the name of the excel file
                control_A_data_origin.append(A_origin[A_dataID])
                control_B_data_origin.append(B_origin[B_dataID])

            # use this for the loops later on
            control_data_len = len(control_data)

            # there's no point in not having data ...
            # ... or something went wrong. rather stop here
            if not control_data_len > 0:
                raise IOError('no data loaded')

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
        sis_log_Both = kwargs.get('sis_log_Both', None)
        non_sis_log_Both = kwargs.get('non_sis_log_Both', None)
        log_bool = (sis_log_Both is not None) and (non_sis_log_Both is not None)
        if log_bool:
            log_sis_array = np.array([['sis_' + key, df_value, len(df_value)] for key, df_value in sis_log_Both.copy().items()])
            log_non_sis_array = np.array([['nonsis_' + key, df_value, len(df_value)] for key, df_value in non_sis_log_Both.copy().items()])
            log_both_array = np.concatenate((log_sis_array, log_non_sis_array), axis=0)

        # array of the key, dataframe and length of all bacteria in sis and nonsis dataset, and their concatenation
        sis_array = np.array([['sis_' + key, df_value, len(df_value)] for key, df_value in sis_Both.copy().items()])
        non_sis_array = np.array([['nonsis_' + key, df_value, len(df_value)] for key, df_value in non_sis_Both.copy().items()])
        both_array = np.concatenate((sis_array, non_sis_array), axis=0)

        # print(len(both_dict), len(both_array))

        # overlapping_lengths = [ for length in np.unique([l])]
        overlapping_lengths = []
        for length in np.unique(both_array[:, 2]):
            sis_and_non_same_len = (len(np.concatenate((
                np.unique([[name.split('_')[ind] for ind in [0, 2]] for name in sis_array[np.where(abs(length - sis_array[:, 2]) <= difference_criterion)][:, 0]]),
                np.unique([[name.split('_')[ind] for ind in [0, 2]] for name in non_sis_array[np.where(abs(length - non_sis_array[:, 2]) <= difference_criterion)][:, 0]])), axis=0)) > 2)
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
            sis_keys_in_overlapping_lengths_unique = np.unique(
                [[int(s) for s in sis_keys_in_overlapping_lengths[ind].split('_') if s.isdigit()] for ind in range(len(sis_keys_in_overlapping_lengths))])
            print('sis_keys_in_overlapping_lengths_unique\n', sis_keys_in_overlapping_lengths_unique)
            print('if we use two sisters in the same trap vs. in different traps\n', len(sis_keys_in_overlapping_lengths), len(sis_keys_in_overlapping_lengths_unique))
            percentage_of_sis_trajs_in_overlapping_lengths = len(sis_keys_in_overlapping_lengths_unique) / int(len(sis_Both) / 2)
            print('percentage_of_sis_trajs_in_overlapping_lengths\n', percentage_of_sis_trajs_in_overlapping_lengths)

            non_sis_keys_in_overlapping_lengths = [key for key, length in zip(non_sis_array[:, 0], non_sis_array[:, 2]) if length in overlapping_lengths]
            print('non_sis_keys_in_overlapping_lengths\n', non_sis_keys_in_overlapping_lengths)
            non_sis_keys_in_overlapping_lengths_unique = np.unique(
                [[int(s) for s in non_sis_keys_in_overlapping_lengths[ind].split('_') if s.isdigit()] for ind in range(len(non_sis_keys_in_overlapping_lengths))])
            print('non_sis_keys_in_overlapping_lengths_unique\n', non_sis_keys_in_overlapping_lengths_unique)
            print('if we use two non_sisters in the same trap vs. in different traps\n', len(non_sis_keys_in_overlapping_lengths), len(non_sis_keys_in_overlapping_lengths_unique))
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
        log_A_dict = dict()
        log_B_dict = dict()
        log_Both_dict = dict()
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
            first_key = random.choice(keys_in_overlapping_lengths)

            keys_that_are_not_related = [key for key in keys_in_overlapping_lengths if not (key.split('_')[0] == first_key.split('_')[0] and key.split('_')[2] == first_key.split('_')[2])]

            # because sometimes it chooses lengths that give incompatible dataframes
            while (not keys_in_overlapping_lengths) and (not keys_that_are_not_related):
                if (not keys_in_overlapping_lengths):
                    print('keys_in_overlapping_lengths was empty, choosing another length')
                if (not keys_that_are_not_related):
                    print('no unrelated keys found!')
                # choose from said distribution of generation lengths, with replacement (better without replacement)
                length[0] = np.random.choice(overlapping_lengths, size=1, replace=True, p=weights)
                keys_in_overlapping_lengths = [key for key, lengthy in zip(both_array[:, 0], both_array[:, 2]) if (abs(lengthy - length) <= difference_criterion) and (key not in repetitions)]
                first_key = random.choice(keys_in_overlapping_lengths)
                keys_that_are_not_related = [key for key in keys_in_overlapping_lengths if not (key.split('_')[0] == first_key.split('_')[0] and key.split('_')[2] == first_key.split('_')[2])]

            second_key = random.choice(keys_that_are_not_related)

            # sanity check to make sure they are not related
            if first_key.split('_')[0] == second_key.split('_')[0] and first_key.split('_')[2] == second_key.split('_')[2]:
                print('chose cells in the same trap!')
                exit()

            # update the returning dictionaries
            Both_dict.update(
                {'A_' + str(new_order): both_array[np.where(first_key == both_array[:, 0])][:, 1][0], 'B_' + str(new_order): both_array[np.where(second_key == both_array[:, 0])][:, 1][0]})
            A_dict.update({str(new_order): both_array[np.where(first_key == both_array[:, 0])][:, 1][0]})
            B_dict.update({str(new_order): both_array[np.where(second_key == both_array[:, 0])][:, 1][0]})
            if log_bool:
                log_Both_dict.update(
                    {'A_' + str(new_order): log_both_array[np.where(first_key == both_array[:, 0])][:, 1][0], 'B_' + str(new_order): log_both_array[np.where(second_key == both_array[:, 0])][:, 1][0]})
                log_A_dict.update({str(new_order): log_both_array[np.where(first_key == both_array[:, 0])][:, 1][0]})
                log_B_dict.update({str(new_order): log_both_array[np.where(second_key == both_array[:, 0])][:, 1][0]})
            reference_A_dict.update({str(new_order): first_key})
            reference_B_dict.update({str(new_order): second_key})
            # go on to the next dataID (new_order)
            new_order = new_order + 1
            repetitions.append(first_key)
            repetitions.append(second_key)

        return A_dict, B_dict, Both_dict, log_A_dict, log_B_dict, log_Both_dict, reference_A_dict, reference_B_dict

    """ Creates the A and B trajectory dictionaries, as well as general dictionary with both the A and B dictionary elements inside, that contain the 
    trajectory generation dataframes as values and the integers as keys for A and B, and 'A_'+integer/'B_'+integer for the general dictionary """

    def create_dictionaries_of_traces(self, data_len, discretization_variable, _keylist, _data, what_to_subtract, start_index, end_index,
            **kwargs):  # what_to_subtract can be either "trap/traj mean/median" or None

        """  """

        def moving_window_growth_rate(dataID, _data, discretize_by='length', fit_the_length_at_birth=True, moving_window=3):
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
                ret_ks = {'length_birth': [], 'length_births': [], 'growth_rates': [], 'length_final': [], 'length_finals': []}

                # Number of raw data points per generation/cycle
                data_points_per_cycle = np.rint(cycle_durations / .05) + np.ones_like(cycle_durations)

                # The x and y values per generation for the regression to get the growth rate
                domains_for_regression_per_gen = [np.linspace(_data[dataID]['time' + ks][start], _data[dataID]['time' + ks][end], num=num_of_data_points) for start, end, num_of_data_points in
                                                  zip(start_indices, end_indices, data_points_per_cycle)]
                ranges_for_regression_per_gen = [np.log(_data[dataID][discretize_by + ks][start:end + 1])  # the end+1 is due to indexing
                                                 for start, end in zip(start_indices, end_indices)]

                # Arrays where the intercepts (the length at birth) and the slopes (the growth rate) will be stored
                regression_intercepts_per_gen = []
                regression_slopes_per_gen = []
                for domain, y_vals in zip(domains_for_regression_per_gen, ranges_for_regression_per_gen):  # loop over generations in the trace
                    # reshape the x and y values
                    domain = np.array(domain).reshape(-1, 1)
                    y_vals = np.array(y_vals).reshape(-1, 1)


                    # do the regression for the normal growth rate
                    reg = LinearRegression().fit(domain, y_vals)
                    # save them to their respective arrays
                    regression_slopes_per_gen.append(reg.coef_[0][0])
                    regression_intercepts_per_gen.append(np.exp(reg.predict(domain[0].reshape(-1, 1))[0][0]))

                    dictionary_of_variables = {'growth_rates': [], 'length_births': [], 'length_finals': []}

                    # If the trap has less generations than the amount requested in the moving window
                    if moving_window < domain.shape[0]:
                        # loop over all the windows inside the generation
                        for start in range(domain.shape[0] - moving_window):

                            # do the regression
                            reg = LinearRegression().fit(domain[start:start+moving_window, 0].reshape(-1, 1), y_vals[start:start + moving_window, 0].reshape(-1, 1))

                            if reg.coef_[0][0] < 0:
                                # print('growth rate lower than 0:', reg.coef_[0][0], dataID)
                                pass
                            elif np.exp(domain[start, 0]) > np.exp(domain[start + moving_window, 0]):
                                # print('length birth is higher than length_final')
                                pass
                            else:
                                # save them to their respective arrays
                                dictionary_of_variables['growth_rates'].append(reg.coef_[0][0])
                                dictionary_of_variables['length_births'].append(np.exp(domain[start, 0]))
                                dictionary_of_variables['length_finals'].append(np.exp(domain[start + moving_window, 0]))
                    else:
                        continue

                    # update the other dictionary
                    ret_ks['growth_rates'].append(np.array(dictionary_of_variables['growth_rates']))
                    ret_ks['length_births'].append(dictionary_of_variables['length_births'])
                    ret_ks['length_finals'].append(dictionary_of_variables['length_finals'])


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
                    checking_pos = np.array(
                        [length_birth < length_final for length_birth, length_final in zip(regression_intercepts_per_gen, np.array(_data[dataID][discretize_by + ks][end_indices]))])
                else:
                    checking_pos = np.array([length_birth < length_final for length_birth, length_final in
                                             zip(np.array(_data[dataID][discretize_by + ks][start_indices]), np.array(_data[dataID][discretize_by + ks][end_indices]))])
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

                # Duration of growth before division
                ret_ks['generationtime'] = np.around(cycle_durations, decimals=2)

                # Due to limitations of data, mainly that after some obvious division points the length of the bacteria drops, which means that it is
                # shrinking, which we assume is impossible. Either way we present is as an option, just in case.
                if fit_the_length_at_birth:
                    ret_ks['length_birth'] = regression_intercepts_per_gen
                else:
                    ret_ks['length_birth'] = np.array(_data[dataID][discretize_by + ks][start_indices])

                # The measured final length before division, however it is worth to note that we have a very good approximation of this observable by
                # the mapping presented in Susman et al. (2018)
                ret_ks['length_final'] = np.array(_data[dataID][discretize_by + ks][end_indices])

                # The rate at which the bacteria grows in this cycle. NOTICE THAT THIS CHANGED FROM 'growth_length'
                ret_ks['growth_rate'] = regression_slopes_per_gen

                # The fold growth, ie. how much it grew. Defined by the rate at which it grew multiplied by the time it grew for
                # NOTE: We call this 'phi' before, in an earlier version of the code
                ret_ks['fold_growth'] = ret_ks['generationtime'] * ret_ks['growth_rate']

                ret_ks['growth_rates'] = np.array(ret_ks['growth_rates'])

                # The fold growth, ie. how much it grew. Defined by the rate at which it grew multiplied by the time it grew for
                # NOTE: We call this 'phi' before, in an earlier version of the code
                ret_ks['fold_growths'] = (.05 * moving_window) * ret_ks['growth_rates']

                # Calculating the division ratios, percentage of mother length that went to daughter
                div_rats = []
                for final, next_beg in zip(ret_ks['length_final'][:-1], ret_ks['length_birth'][1:]):
                    div_rats.append(next_beg / final)

                # we use the length at birth that is not in the dataframe in order to get enough division ratios
                div_rats.append(_data[dataID][discretize_by + ks][end_indices[-1] + 1] / ret_ks['length_final'][-1])
                ret_ks['division_ratio'] = div_rats

                # the added length to check the adder model
                ret_ks['added_length'] = ret_ks['length_final'] - ret_ks['length_birth']

                normal_length = len(ret_ks['generationtime'])

                if (len(ret_ks['growth_rate']) != normal_length) or (len(ret_ks['length_birth']) != normal_length) or (len(ret_ks['length_final']) != normal_length) or (
                        len(ret_ks['added_length']) != normal_length) or (len(ret_ks['division_ratio']) != normal_length) or (len(ret_ks['fold_growth']) != normal_length):
                    print('Error in the lengths!', (len(ret_ks['growth_rate']) != normal_length), (len(ret_ks['length_birth']) != normal_length), (len(ret_ks['length_final']) != normal_length),
                        (len(ret_ks['added_length']) != normal_length), (len(ret_ks['division_ratio']) != normal_length), (len(ret_ks['fold_growth']) != normal_length))
                    exit()

                # we have everything, now make a dataframe
                ret.append(ret_ks)

            return ret

        # def moving_window_growth_rate(dataID, _data, discretize_by='length', moving_window=3):
        #     # arbitrary but necessary specification options for the two cells inside the trap
        #     keysuffix = ['A', 'B']
        #
        #     # Return two pandas-dataframes for the two trajectories usually contained in one file as two elements of a list.
        #     # As the two sisters do not need to have the same number of cell divisions,
        #     # a single dataframe might cause problems with variably lengthed trajectories.
        #     ret = list()
        #
        #     for ks in keysuffix:
        #
        #         # use this to get the starting and ending indices in the raw data
        #         start_indices, end_indices = self.get_division_indices(data_id=dataID, _data=_data, discretize_by=discretize_by, ks=ks)
        #
        #         # How long each cycle is
        #         cycle_durations = np.array(_data[dataID]['time' + ks][end_indices]) - np.array(_data[dataID]['time' + ks][start_indices])
        #
        #         # Store results in this dictionary, which can be easier transformed into pandas-dataframe
        #         ret_ks = {'length_birth': [], 'length_births': [], 'growth_rates': [], 'length_final': [], 'length_finals': []}
        #
        #         # Number of raw data points per generation/cycle
        #         data_points_per_cycle = np.rint(cycle_durations / .05) + np.ones_like(cycle_durations)
        #
        #         # The x and y values per generation for the regression to get the growth rate
        #         domains_for_regression_per_gen = [np.linspace(_data[dataID]['time' + ks][start], _data[dataID]['time' + ks][end], num=num_of_data_points) for start, end, num_of_data_points in
        #                                           zip(start_indices, end_indices, data_points_per_cycle)]
        #         ranges_for_regression_per_gen = [np.log(_data[dataID][discretize_by + ks][start:end + 1])  # the end+1 is due to indexing
        #                                          for start, end in zip(start_indices, end_indices)]
        #
        #         for domain, y_vals in zip(domains_for_regression_per_gen, ranges_for_regression_per_gen):  # loop over generations in the trace
        #             # reshape the x and y values
        #             domain = np.array(domain).reshape(-1, 1)
        #             y_vals = np.array(y_vals).reshape(-1, 1)
        #
        #             dictionary_of_variables = {'growth_rates': [], 'length_births': [], 'length_finals': []}
        #
        #             # If the trap has less generations than the amount requested in the moving window
        #             if moving_window < domain.shape[0]:
        #                 # loop over all the windows inside the generation
        #                 for start in range(domain.shape[0] - moving_window):
        #
        #                     # do the regression
        #                     reg = LinearRegression().fit(domain[start:start+moving_window, 0].reshape(-1, 1), y_vals[start:start + moving_window, 0].reshape(-1, 1))
        #
        #                     if reg.coef_[0][0] < 0:
        #                         # print('growth rate lower than 0:', reg.coef_[0][0], dataID)
        #                         pass
        #                     elif np.exp(domain[start, 0]) > np.exp(domain[start + moving_window, 0]):
        #                         # print('length birth is higher than length_final')
        #                         pass
        #                     else:
        #                         # save them to their respective arrays
        #                         dictionary_of_variables['growth_rates'].append(reg.coef_[0][0])
        #                         dictionary_of_variables['length_births'].append(np.exp(domain[start, 0]))
        #                         dictionary_of_variables['length_finals'].append(np.exp(domain[start + moving_window, 0]))
        #             else:
        #                 continue
        #
        #             # update the other dictionary
        #             ret_ks['length_birth'].append(np.exp(LinearRegression().fit(domain, y_vals).predict(domain[0].reshape(-1, 1)))[0][0])
        #             ret_ks['length_final'].append(np.exp(LinearRegression().fit(domain, y_vals).predict(domain[-1].reshape(-1, 1)))[0][0])
        #             ret_ks['growth_rates'].append(np.array(dictionary_of_variables['growth_rates']))
        #             ret_ks['length_births'].append(dictionary_of_variables['length_births'])
        #             ret_ks['length_finals'].append(dictionary_of_variables['length_finals'])
        #
        #         # Duration of growth before division
        #         ret_ks['generationtime'] = np.around(cycle_durations, decimals=2)
        #
        #         # ret_ks['length_final'] = np.array(_data[dataID][discretize_by + ks][end_indices])
        #
        #         ret_ks['length_final'] = np.array(ret_ks['length_final'])
        #
        #         ret_ks['length_birth'] = np.array(ret_ks['length_birth'])
        #
        #         ret_ks['growth_rates'] = np.array(ret_ks['growth_rates'])
        #
        #         # The fold growth, ie. how much it grew. Defined by the rate at which it grew multiplied by the time it grew for
        #         # NOTE: We call this 'phi' before, in an earlier version of the code
        #         ret_ks['fold_growths'] = (.05 * moving_window) * ret_ks['growth_rates']
        #
        #         # Calculating the division ratios, percentage of mother length that went to daughter
        #         div_rats = []
        #         for final, next_beg in zip(ret_ks['length_final'][:-1], ret_ks['length_birth'][1:]):
        #             div_rats.append(next_beg / final)
        #
        #         # we use the length at birth that is not in the dataframe in order to get enough division ratios
        #         div_rats.append(_data[dataID][discretize_by + ks][end_indices[-1] + 1] / ret_ks['length_final'][-1])
        #         ret_ks['division_ratio'] = div_rats
        #
        #         # the added length to check the adder model
        #         ret_ks['added_length'] = ret_ks['length_final'] - ret_ks['length_birth']
        #
        #         # we have everything, now make a dataframe
        #         ret.append(ret_ks)
        #
        #     return ret

        """ Creates a list of two pandas dataframes, from a trap's two trajectories, that contain the values for the variables in each generation """

        def generation_dataframe_creation(dataID, _data, discretize_by='length', fit_the_length_at_birth=True, moving_window=3):

            def get_the_inside_generation_dictionary(dataID, _data, cycle_durations, start_indices, end_indices, ret_ks, discretize_by='length', moving_window=3):

                # Store results in this dictionary, which can be easier transformed into pandas-dataframe
                inside_gen_dict = {'length_births': [], 'growth_rates': [], 'fold_growths': [], 'length_finals': []}
                inside_gen_dict.update(dict(zip(ret_ks.keys(), [[] for key in ret_ks.keys()])))

                # Number of raw data points per generation/cycle
                data_points_per_cycle = np.rint(cycle_durations / .05) + np.ones_like(cycle_durations)

                # The x and y values per generation for the regression to get the growth rate
                domains_for_regression_per_gen = [np.linspace(_data[dataID]['time' + ks][start], _data[dataID]['time' + ks][end], num=num_of_data_points) for start, end, num_of_data_points in
                                                  zip(start_indices, end_indices, data_points_per_cycle)]
                ranges_for_regression_per_gen = [np.log(_data[dataID][discretize_by + ks][start:end + 1])  # the end+1 is due to indexing
                                                 for start, end in zip(start_indices, end_indices)]
                
                for domain, y_vals, gen_ind in zip(domains_for_regression_per_gen, ranges_for_regression_per_gen, np.arange(len(ranges_for_regression_per_gen))):  # loop over generations in the trace
                    # reshape the x and y values
                    domain = np.array(domain).reshape(-1, 1)
                    y_vals = np.array(y_vals).reshape(-1, 1)

                    dictionary_of_variables = {'growth_rates': [], 'length_births': [], 'length_finals': [], 'fold_growths': []}

                    # If the trap has less generations than the amount requested in the moving window
                    if moving_window < domain.shape[0]:
                        # loop over all the windows inside the generation
                        for start in range(domain.shape[0] - moving_window):

                            # do the regression
                            reg = LinearRegression().fit(domain[start:start + moving_window, 0].reshape(-1, 1), y_vals[start:start + moving_window, 0].reshape(-1, 1))

                            if reg.coef_[0][0] < 0:
                                # print('growth rate lower than 0:', reg.coef_[0][0], dataID)
                                pass
                            elif np.exp(domain[start, 0]) > np.exp(domain[start + moving_window, 0]):
                                # print('length birth is higher than length_final')
                                pass
                            else:
                                # save them to their respective arrays
                                dictionary_of_variables['growth_rates'].append(reg.coef_[0][0])
                                dictionary_of_variables['length_births'].append(np.exp(domain[start, 0]))
                                dictionary_of_variables['length_finals'].append(np.exp(domain[start + moving_window, 0]))
                                dictionary_of_variables['fold_growths'].append((.05 * moving_window) * reg.coef_[0][0])

                            include_outside_variables = True
                    else:
                        include_outside_variables = False
                        print('something else happened to continue')
                        continue

                    # update the other dictionary
                    inside_gen_dict['growth_rates'].append(np.array(dictionary_of_variables['growth_rates']))
                    inside_gen_dict['length_births'].append(np.array(dictionary_of_variables['length_births']))
                    inside_gen_dict['length_finals'].append(np.array(dictionary_of_variables['length_finals']))
                    inside_gen_dict['fold_growths'].append(np.array(dictionary_of_variables['fold_growths']))

                    if include_outside_variables:
                        for key, val in ret_ks.items():
                            inside_gen_dict[key].append(val[gen_ind])
                    else:
                        print('should go with "something else happened to continue"')
                        pass

                # For comparison
                len_of_growth_rates = len(inside_gen_dict['growth_rates'])

                # Checking
                if (len(inside_gen_dict['length_births']) != len_of_growth_rates) or (len(inside_gen_dict['length_finals']) != len_of_growth_rates) or (
                        len(inside_gen_dict['fold_growths']) != len_of_growth_rates):
                    print('ERROR the inside variables are not the same size',
                        (len(inside_gen_dict['length_births']) != len_of_growth_rates) or (len(inside_gen_dict['length_finals']) != len_of_growth_rates) or (
                                    len(inside_gen_dict['fold_growths']) != len_of_growth_rates))
                    exit()

                # Checking
                for key in ret_ks.keys():
                    if len(inside_gen_dict[key]) != len_of_growth_rates:
                        print('ERROR the cycle variables are not the same size', key, len_of_growth_rates, len(inside_gen_dict[key]))
                        print(len(cycle_durations), len(domains_for_regression_per_gen), len(ranges_for_regression_per_gen), len(start_indices), len(end_indices))
                        exit()
                
                return inside_gen_dict

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
                domains_for_regression_per_gen = [np.linspace(_data[dataID]['time' + ks][start], _data[dataID]['time' + ks][end], num=num_of_data_points) for start, end, num_of_data_points in
                                                  zip(start_indices, end_indices, data_points_per_cycle)]
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
                    checking_pos = np.array(
                        [length_birth < length_final for length_birth, length_final in zip(regression_intercepts_per_gen, np.array(_data[dataID][discretize_by + ks][end_indices]))])
                else:
                    checking_pos = np.array([length_birth < length_final for length_birth, length_final in
                                             zip(np.array(_data[dataID][discretize_by + ks][start_indices]), np.array(_data[dataID][discretize_by + ks][end_indices]))])
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

                # Duration of growth before division
                ret_ks['generationtime'] = np.around(cycle_durations, decimals=2)

                # Due to limitations of data, mainly that after some obvious division points the length of the bacteria drops, which means that it is
                # shrinking, which we assume is impossible. Either way we present is as an option, just in case.
                if fit_the_length_at_birth:
                    ret_ks['length_birth'] = regression_intercepts_per_gen
                else:
                    ret_ks['length_birth'] = np.array(_data[dataID][discretize_by + ks][start_indices])

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

                # get the inside generation dictionary
                inside_gen_dict = get_the_inside_generation_dictionary(dataID, _data, cycle_durations, start_indices, end_indices, ret_ks, discretize_by='length', moving_window=3)

                # # To do the check below
                # ret_ks_df = pd.DataFrame(ret_ks)

                # # checking that the length_final > length_birth
                # if pd.Series(ret_ks_df['length_final'] <= ret_ks_df['length_birth']).isna().any():
                #     print('length_final <= length_birth, and this so-called "generation" was taken out of its dataframe')
                #     ret_ks_df = ret_ks_df.drop(index=np.where(ret_ks_df['length_final'] <= ret_ks_df['length_birth'])[0]).reset_index(drop=True)

                # we have everything, now make a dataframe
                ret.append(pd.DataFrame(ret_ks))
                ret.append(inside_gen_dict)
            return ret


        # """ Creates a list of two pandas dataframes, from a trap's two trajectories, that contain the values for the variables in each generation """
        #
        # def generation_dataframe_creation(dataID, _data, discretize_by='length', fit_the_length_at_birth=True):
        #     """
        #     Transform measured time series data into data for each generation:
        #     (1) Find cell division events as a large enough drop in the measured value given by 'discretize_by'
        #     (2) Compute various observables from these generations of cells
        #     (3) Returns two pandas dataframes for each of the discretized trajectories
        #     """
        #
        #     # arbitrary but necessary specification options for the two cells inside the trap
        #     keysuffix = ['A', 'B']
        #
        #     # Return two pandas-dataframes for the two trajectories usually contained in one file as two elements of a list.
        #     # As the two sisters do not need to have the same number of cell divisions,
        #     # a single dataframe might cause problems with variably lengthed trajectories.
        #     ret = list()
        #
        #     for ks in keysuffix:
        #
        #         # use this to get the starting and ending indices in the raw data
        #         start_indices, end_indices = self.get_division_indices(data_id=dataID, _data=_data, discretize_by=discretize_by, ks=ks)
        #
        #         # How long each cycle is
        #         cycle_durations = np.array(_data[dataID]['time' + ks][end_indices]) - np.array(_data[dataID]['time' + ks][start_indices])
        #
        #         # Store results in this dictionary, which can be easier transformed into pandas-dataframe
        #         ret_ks = dict()
        #
        #         # Number of raw data points per generation/cycle
        #         data_points_per_cycle = np.rint(cycle_durations / .05) + np.ones_like(cycle_durations)
        #
        #         # The x and y values per generation for the regression to get the growth rate
        #         domains_for_regression_per_gen = [np.linspace(_data[dataID]['time' + ks][start], _data[dataID]['time' + ks][end], num=num_of_data_points) for start, end, num_of_data_points in
        #                                           zip(start_indices, end_indices, data_points_per_cycle)]
        #         ranges_for_regression_per_gen = [np.log(_data[dataID][discretize_by + ks][start:end + 1])  # the end+1 is due to indexing
        #                                          for start, end in zip(start_indices, end_indices)]
        #
        #         # Arrays where the intercepts (the length at birth) and the slopes (the growth rate) will be stored
        #         regression_intercepts_per_gen = []
        #         regression_slopes_per_gen = []
        #         for domain, y_vals in zip(domains_for_regression_per_gen, ranges_for_regression_per_gen):  # loop over generations in the trace
        #             # reshape the x and y values
        #             domain = np.array(domain).reshape(-1, 1)
        #             y_vals = np.array(y_vals).reshape(-1, 1)
        #             # do the regression
        #             reg = LinearRegression().fit(domain, y_vals)
        #             # save them to their respective arrays
        #             regression_slopes_per_gen.append(reg.coef_[0][0])
        #             regression_intercepts_per_gen.append(np.exp(reg.predict(domain[0].reshape(-1, 1))[0][0]))
        #
        #         # change them to numpy arrays
        #         regression_slopes_per_gen = np.array(regression_slopes_per_gen)
        #         regression_intercepts_per_gen = np.array(regression_intercepts_per_gen)
        #
        #         # check if the growth_rate is negative, meaning it is obviously a not a generation
        #         checking_pos = np.array([slope > 0 for slope in regression_slopes_per_gen])
        #         if not checking_pos.all():
        #             print("there's been a negative or zero growth_rate found!")
        #             # change growth_rate
        #             regression_slopes_per_gen = regression_slopes_per_gen[np.where(checking_pos)]
        #             # change length at birth
        #             regression_intercepts_per_gen = regression_intercepts_per_gen[np.where(checking_pos)]
        #             # change generationtime
        #             cycle_durations = cycle_durations[np.where(checking_pos)]
        #             # change the indices which will change the rest
        #             start_indices, end_indices = start_indices[np.where(checking_pos)], end_indices[np.where(checking_pos)]
        #
        #         # check if the length_final <= length_birth, meaning it is obviously a not a generation
        #         if fit_the_length_at_birth:
        #             checking_pos = np.array(
        #                 [length_birth < length_final for length_birth, length_final in zip(regression_intercepts_per_gen, np.array(_data[dataID][discretize_by + ks][end_indices]))])
        #         else:
        #             checking_pos = np.array([length_birth < length_final for length_birth, length_final in
        #                                      zip(np.array(_data[dataID][discretize_by + ks][start_indices]), np.array(_data[dataID][discretize_by + ks][end_indices]))])
        #         if not checking_pos.all():
        #             print('length_final <= length_birth, and this so-called "generation" was taken out of its dataframe')
        #             # change growth_rate
        #             regression_slopes_per_gen = regression_slopes_per_gen[np.where(checking_pos)]
        #             # change length at birth
        #             regression_intercepts_per_gen = regression_intercepts_per_gen[np.where(checking_pos)]
        #             # change generationtime
        #             cycle_durations = cycle_durations[np.where(checking_pos)]
        #             # change the indices which will change the rest
        #             start_indices, end_indices = start_indices[np.where(checking_pos)], end_indices[np.where(checking_pos)]
        #
        #         # Duration of growth before division
        #         ret_ks['generationtime'] = np.around(cycle_durations, decimals=2)
        #
        #         # Due to limitations of data, mainly that after some obvious division points the length of the bacteria drops, which means that it is
        #         # shrinking, which we assume is impossible. Either way we present is as an option, just in case.
        #         if fit_the_length_at_birth:
        #             ret_ks['length_birth'] = regression_intercepts_per_gen
        #         else:
        #             ret_ks['length_birth'] = np.array(_data[dataID][discretize_by + ks][start_indices])
        #
        #         # The measured final length before division, however it is worth to note that we have a very good approximation of this observable by
        #         # the mapping presented in Susman et al. (2018)
        #         ret_ks['length_final'] = np.array(_data[dataID][discretize_by + ks][end_indices])
        #
        #         # The rate at which the bacteria grows in this cycle. NOTICE THAT THIS CHANGED FROM 'growth_length'
        #         ret_ks['growth_rate'] = regression_slopes_per_gen
        #
        #         # The fold growth, ie. how much it grew. Defined by the rate at which it grew multiplied by the time it grew for
        #         # NOTE: We call this 'phi' before, in an earlier version of the code
        #         ret_ks['fold_growth'] = ret_ks['generationtime'] * ret_ks['growth_rate']
        #
        #         # Calculating the division ratios, percentage of mother length that went to daughter
        #         div_rats = []
        #         for final, next_beg in zip(ret_ks['length_final'][:-1], ret_ks['length_birth'][1:]):
        #             div_rats.append(next_beg / final)
        #
        #         # we use the length at birth that is not in the dataframe in order to get enough division ratios
        #         div_rats.append(_data[dataID][discretize_by + ks][end_indices[-1] + 1] / ret_ks['length_final'][-1])
        #         ret_ks['division_ratio'] = div_rats
        #
        #         # the added length to check the adder model
        #         ret_ks['added_length'] = ret_ks['length_final'] - ret_ks['length_birth']
        #
        #         # # To do the check below
        #         # ret_ks_df = pd.DataFrame(ret_ks)
        #
        #         # # checking that the length_final > length_birth
        #         # if pd.Series(ret_ks_df['length_final'] <= ret_ks_df['length_birth']).isna().any():
        #         #     print('length_final <= length_birth, and this so-called "generation" was taken out of its dataframe')
        #         #     ret_ks_df = ret_ks_df.drop(index=np.where(ret_ks_df['length_final'] <= ret_ks_df['length_birth'])[0]).reset_index(drop=True)
        #
        #         # we have everything, now make a dataframe
        #         ret.append(pd.DataFrame(ret_ks))
        #     return ret

        def minusing(what_to_subtract, Traj_A, Traj_B, global_df):
            if what_to_subtract == None:
                # we want the measurement, so we do nothing else
                pass
            elif what_to_subtract.split(' ')[1] == 'mean':
                # we want to subtract some mean to each value
                if what_to_subtract.split(' ')[0] == 'trap':
                    trap = pd.concat([Traj_A, Traj_B], axis=0).mean()
                    Traj_A = Traj_A - trap
                    Traj_B = Traj_B - trap
                elif what_to_subtract.split(' ')[0] == 'traj':
                    Traj_A = Traj_A - Traj_A.mean()
                    Traj_B = Traj_B - Traj_B.mean()
                elif (what_to_subtract.split(' ')[0] == 'global') and (global_df is not None):
                    Traj_A = Traj_A - global_df.mean()
                    Traj_B = Traj_B - global_df.mean()
            elif what_to_subtract.split(' ')[1] == 'median':
                # we want to subtract some median to each value
                if what_to_subtract.split(' ')[0] == 'trap':
                    trap = pd.concat([Traj_A, Traj_B], axis=0).median()
                    Traj_A = Traj_A - trap
                    Traj_B = Traj_B - trap
                elif what_to_subtract.split(' ')[0] == 'traj':
                    Traj_A = Traj_A - Traj_A.median()
                    Traj_B = Traj_B - Traj_B.median()
                elif (what_to_subtract.split(' ')[0] == 'global') and (global_df is not None):
                    Traj_A = Traj_A - global_df.median()
                    Traj_B = Traj_B - global_df.median()
            else:
                print('wrong what_to_subtract in generation_dataframe_creation')

            return Traj_A, Traj_B

        # Only not None when we want global mean/median
        global_df = kwargs.get('global_df', None)
        log_global_df = kwargs.get('log_global_df', None)

        # creation of the dictionaries
        Both_dict = dict()
        A_dict = dict()
        B_dict = dict()
        log_Both_dict = dict()
        log_A_dict = dict()
        log_B_dict = dict()
        inside_generation_A_dict = dict()
        inside_generation_B_dict = dict()

        # loop over the amount of data in the subclass
        for experiment_number in range(data_len):
            # get the trajectory generation dataframe
            TrajA, TrajA_finer, TrajB, TrajB_finer = generation_dataframe_creation(dataID=experiment_number, _data=_data, discretize_by=discretization_variable, fit_the_length_at_birth=True)

            # TrajA_finer, TrajB_finer = moving_window_growth_rate(dataID=experiment_number, _data=_data, discretize_by=discretization_variable, fit_the_length_at_birth=True, moving_window=3)

            # apply the hard generation limits across all traces
            TrajA = TrajA.iloc[start_index:end_index]
            TrajB = TrajB.iloc[start_index:end_index]

            log_TrajA = np.log(TrajA.iloc[start_index:end_index])
            log_TrajB = np.log(TrajB.iloc[start_index:end_index])

            TrajA, TrajB = minusing(what_to_subtract=what_to_subtract, Traj_A=TrajA, Traj_B=TrajB, global_df=global_df)
            log_TrajA, log_TrajB = minusing(what_to_subtract=what_to_subtract, Traj_A=log_TrajA, Traj_B=log_TrajB, global_df=log_global_df)

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
            inside_generation_A_dict.update({str(experiment_number): TrajA_finer})
            inside_generation_B_dict.update({str(experiment_number): TrajB_finer})

        return Both_dict, A_dict, B_dict, log_Both_dict, log_A_dict, log_B_dict, inside_generation_A_dict, inside_generation_B_dict

    """ Here we create the mother and daughter dataframe that contains all the instances of mother and daughter across Sister and NS datasets.
                This is only meant for the Population subclass because we are not differentiating between A/B.
                We can choose between the normalized lengths dictionary or the units one. """

    def intergenerational_dataframe_creations(self, all_data_dict, how_many_separations):

        # These are where we will put the older and newer generations respectively
        mother_dfs = []
        daughter_dfs = []

        # 1 separation is mother daughter, 2 is grandmother granddaughter, etc...
        for separation in range(1, how_many_separations + 1):
            # the dataframe for the older and the younger generations
            older_df = pd.DataFrame(columns=self._relationship_dfs_variable_names)
            younger_df = pd.DataFrame(columns=self._relationship_dfs_variable_names)

            # loop over every key in both S and NS datasets
            for keyA, keyB in zip(list(all_data_dict.keys())[:-1:2], list(all_data_dict.keys())[1::2]):
                # concatenate both the A and B trajectory in that order
                both_traj_df = pd.concat([all_data_dict[keyA].copy(), all_data_dict[keyB].copy()]).reset_index(drop=True)

                # decides which generation to put so we don't have any conflicting or non-true inter-generational pairs
                older_mask = [ind for ind in range(len(all_data_dict[keyA]) - separation)] + [len(all_data_dict[keyA]) + ind for ind in range(len(all_data_dict[keyB]) - separation)]
                younger_mask = [separation + ind for ind in range(len(all_data_dict[keyA]) - separation)] + [separation + len(all_data_dict[keyA]) + ind for ind in
                                                                                                             range(len(all_data_dict[keyB]) - separation)]

                # add this trap's mother and daughter cells to the df
                older_df = pd.concat([older_df, both_traj_df.iloc[older_mask]], axis=0, join='inner').reset_index(drop=True)
                younger_df = pd.concat([younger_df, both_traj_df.iloc[younger_mask]], axis=0, join='inner').reset_index(drop=True)

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
            A_df = pd.DataFrame(columns=self._variable_names)
            B_df = pd.DataFrame(columns=self._variable_names)

            # Because it is not a given that all the experiments done will have a this many generations recorded
            A_keys_with_this_length = [keyA for keyA, keyB in zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[keyA]), len(B_dict[keyB])) > generation + 1]
            B_keys_with_this_length = [keyB for keyA, keyB in zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[keyA]), len(B_dict[keyB])) > generation + 1]

            # looping over all pairs recorded, ie. all traps/experiments
            for keyA, keyB in zip(A_keys_with_this_length, B_keys_with_this_length):
                # add the data, trap means and traj means to the dataframe that collects them for all traps/experiments
                A_df = pd.concat([A_df, A_dict[keyA].iloc[generation].to_frame().T.reset_index(drop=True)], axis=0)
                B_df = pd.concat([B_df, B_dict[keyB].iloc[generation].to_frame().T.reset_index(drop=True)], axis=0)

            # reset the index because it is convinient and they were all added with an index 0 since we are comparing sisters
            A_df = A_df.reset_index(drop=True)
            B_df = B_df.reset_index(drop=True)

            # add it to the array that contains the sisters, first cousins, second cousins, etc...
            A_df_array.append(A_df)
            B_df_array.append(B_df)

        return A_df_array, B_df_array



    """ END of Main Definitions """

    """ This is made to have as a table to look at for reference as to what functions we can use on an instance and what attributes we can use """

    def table_with_attributes_and_available_methods(self):
        method_list = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]
        dictionary = {'Class Name': self.__class__.__name__, 'Attributes': list(self.__dict__.keys()), 'Methods': method_list}
        return dictionary
    
    """ pool together all the traces in a dataset """
    
    def get_pooled_trap_dict(self, A_dict, B_dict, variable_names):
        df_A = pd.DataFrame(columns=variable_names, dtype=float)
        df_B = pd.DataFrame(columns=variable_names, dtype=float)
        for val_A, val_B in zip(A_dict.values(), B_dict.values()):
            min_len = min(len(val_A), len(val_B))
            df_A = df_A.append(val_A.iloc[:min_len])
            df_B = df_B.append(val_B.iloc[:min_len])

        return df_A.reset_index(drop=True), df_B.reset_index(drop=True)
    
    # """ input a dataframe and this gives back the bin number of each entry """
    #
    # def put_the_bins_on_side_categories(self, A_df, B_df, log_A_df, log_B_df, bins_on_side, variable_names, log_vars):
    #     df_A_cut = pd.DataFrame(columns=variable_names)
    #     df_B_cut = pd.DataFrame(columns=variable_names)
    #     for var in variable_names:
    #         if var in log_vars:
    #             A_centered = log_A_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]
    #             B_centered = log_B_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]
    #         else:
    #             A_centered = A_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]
    #             B_centered = B_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]
    #         joint_centered = pd.concat([A_centered, B_centered]).reset_index(drop=True)
    #
    #         edges = np.append(np.linspace(np.min(joint_centered), 0, bins_on_side+1), np.linspace(0, np.max(joint_centered), bins_on_side+1)[1:])
    #
    #         for indexx in range(len(edges) - 1):
    #             if edges[indexx] >= edges[indexx + 1]:
    #                 print('edges:', edges)
    #                 print(joint_centered)
    #
    #         # based on the bin edge, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
    #         A_cut = pd.cut(A_centered, edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))  # , duplicates='drop'
    #         B_cut = pd.cut(B_centered, edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))
    #
    #         if A_cut.isna().any():
    #             print('_____')
    #             print(var)
    #             print('A')
    #             # print(A_cut.isna().sum())
    #
    #             if (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(edges)[0])[0] < .0001):
    #                 print('its close to the left one!')
    #                 A_cut.iloc[np.where(A_cut.isna())] = 0.0  # print(A_cut.isna().any())
    #             elif (abs(np.array(A_centered.iloc[np.where(A_cut.isna())]) - np.array(edges)[-1])[0] < .0001):
    #                 print('its close to the right one!')
    #                 A_cut.iloc[np.where(A_cut.isna())] = np.int64(len(edges) - 2)  # print(A_cut.isna().any())
    #
    #         if B_cut.isna().any():
    #             print('_____')
    #             print(var)
    #             print('B')
    #             # print(B_cut.isna().sum())
    #
    #             if (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edges)[0])[0] < .0001):
    #                 print('its close to the left one!')
    #                 B_cut.iloc[np.where(B_cut.isna())] = 0.0  # print(B_cut.isna().any())
    #             elif (abs(np.array(B_centered.iloc[np.where(B_cut.isna())]) - np.array(edges)[-1])[0] < .0001):
    #                 print('its close to the right one!')
    #                 B_cut.iloc[np.where(B_cut.isna())] = np.int64(len(edges) - 2)  # print(B_cut.isna().any())
    #
    #         # append this trap's variable's labels
    #         df_A_cut[var] = A_cut
    #         df_B_cut[var] = B_cut
    #
    #     # checking there are still no nans
    #     if df_A_cut.isna().values.any():
    #         print('df_A_cut has nans still!')
    #         exit()
    #     if df_B_cut.isna().values.any():
    #         print('df_B_cut has nans still!')
    #         exit()
    #
    #     return df_A_cut, df_B_cut
    #
    # """ using the 'kmeans' strategy in KBinsDiscretizer to discretize distribution into n_bins """
    #
    # def discrete_binning_method_kmeans(self, A_df, B_df, log_A_df, log_B_df, bins_in_total, variable_names, log_vars):
    #     df_A_cut = pd.DataFrame(columns=variable_names)
    #     df_B_cut = pd.DataFrame(columns=variable_names)
    #     for var in variable_names:
    #         if var in log_vars:
    #             A_centered = np.array(log_A_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]).reshape(-1, 1)
    #             B_centered = np.array(log_B_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]).reshape(-1, 1)
    #         else:
    #             A_centered = np.array(A_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]).reshape(-1, 1)
    #             B_centered = np.array(B_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]).reshape(-1, 1)
    #         joint_centered = np.concatenate((A_centered, B_centered), axis=0)
    #
    #         est = KBinsDiscretizer(n_bins=bins_in_total, encode='ordinal', strategy='kmeans')
    #         est.fit(joint_centered)
    #
    #         edges = np.sort(est.bin_edges_[0])
    #
    #         # based on the bin edge, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
    #         A_cut = pd.cut(A_centered.flatten(), edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))  # , duplicates='drop'
    #         B_cut = pd.cut(B_centered.flatten(), edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))
    #
    #         print('number of bins:', est.n_bins)
    #         if est.n_bins != bins_in_total:
    #             raise IOError('est.n_bins != bins_in_total')
    #
    #         # append this trap's variable's labels
    #         df_A_cut[var] = A_cut
    #         df_B_cut[var] = B_cut
    #
    #     return df_A_cut, df_B_cut
    #
    #
    # """ This gives us the joint probabilities for all variable combinations """
    #
    # def get_joint_probs_dict(self, A_categories, B_categories, variable_names, bins_on_side, base_of_log, precision):
    #     joint_probs_dict = dict()
    #     for var1 in variable_names:
    #         for var2 in variable_names:
    #             joint_probs_dict.update({'A__' + var1 + '__' + 'B__' + var2:
    #                                          self.get_entropies_and_joint_probs_from_labeled_data(A_cut=A_categories[var1], B_cut=B_categories[var2], bins_on_side=bins_on_side,
    #                                              base_of_log=base_of_log, precision=precision)[3]})
    #
    #     return joint_probs_dict
    #
    # """ this saves the joint distribution heatmaps """
    #
    # def show_the_joint_dists(self, joint_probs, A_vars, B_vars, bins_on_side, mean, dataset, directory, precision):
    #     filename = dataset+' joint probabilities'
    #     A_syms = dict(zip(['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio', 'added_length'],
    #         [r'$\tau_A$', r'$x_A(0)$', r'$x_A(\tau)$', r'$\alpha_A$', r'$\phi_A$', r'$f_A$', r'$\Delta_A$']))
    #     B_syms = dict(zip(['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio', 'added_length'],
    #         [r'$\tau_B$', r'$x_B(0)$', r'$x_B(\tau)$', r'$\alpha_B$', r'$\phi_B$', r'$f_B$', r'$\Delta_B$']))
    #
    #     heatmaps = [[key, val] for key, val in joint_probs.items() if (key.split('__')[1] in A_vars) and (key.split('__')[3] in B_vars)]
    #
    #     fig, axes = plt.subplots(ncols=len(A_vars), nrows=len(B_vars), sharey='row', sharex='col', figsize=(12.7, 7.5))
    #     index = 0
    #     for ax in axes.flatten():
    #         key = heatmaps[index][0].split('__')
    #         df = pd.DataFrame(columns=np.arange(2 * bins_on_side), index=np.arange(2 * bins_on_side), dtype=float)
    #         for col in df.columns:
    #             for ind in df.index:
    #                 df[col].loc[ind] = heatmaps[index][1]['{}_{}'.format(col, ind)]
    #         sns.heatmap(data=df, ax=ax, annot=True, vmin=0, vmax=1, cbar=False, fmt='.{}f'.format(precision))  # xticklabels=np.arange(bins_on_side), yticklabels=np.arange(bins_on_side)
    #         # ax.set_title(A_syms[key[1]]+' '+B_syms[key[3]])
    #         index += 1
    #
    #     for ax, row in zip(axes[:, 0], [A_syms[heatmaps[ind][0].split('__')[1]] for ind in np.arange(0, index, len(B_vars))]):
    #         ax.set_ylabel(row, rotation=0, size='large')
    #
    #     for ax, col in zip(axes[0], [B_syms[heatmaps[ind][0].split('__')[3]] for ind in range(index)]):
    #         ax.set_title(col)
    #
    #     plt.suptitle(filename)
    #     plt.tight_layout(pad=.3, rect=(0, 0, 1, .96))  # rect=(0, 0, 1, .97) rect=(0, 0.03, 1, .97),
    #     # plt.show()
    #     plt.savefig(directory + '/' + filename, dpi=300)
    #     plt.close()
    #
    # """ labeled data entropy """
    #
    # def get_entropies_and_joint_probs_from_labeled_data(self, A_cut, B_cut, bins_on_side, base_of_log, precision):
    #     joint_centered = pd.DataFrame({'A': A_cut, 'B': B_cut})
    #
    #     joint_prob_list = dict([('{}_{}'.format(label_A, label_B), len(joint_centered[(joint_centered['A'] == label_A) & (joint_centered['B'] == label_B)]) / len(joint_centered)) for label_B in
    #                             np.arange(2 * bins_on_side) for label_A in np.arange(2 * bins_on_side)])
    #     A_trace_marginal_probs = dict([('{}'.format(label_A), len(A_cut.iloc[np.where(A_cut == label_A)]) / len(A_cut)) for label_A in np.arange(2 * bins_on_side)])
    #     B_trace_marginal_probs = dict([('{}'.format(label_B), len(B_cut.iloc[np.where(B_cut == label_B)]) / len(B_cut)) for label_B in np.arange(2 * bins_on_side)])
    #
    #     # conditioning the A trace based on the B trace
    #     A_conditioned_on_B_entropy = np.array([- joint_prob_list[key] * np.log(joint_prob_list[key] / A_trace_marginal_probs[key.split('_')[0]]) for key in joint_prob_list.keys() if
    #                                            joint_prob_list[key] != 0 and A_trace_marginal_probs[key.split('_')[0]] != 0]).sum()
    #     B_conditioned_on_A_entropy = np.array([- joint_prob_list[key] * np.log(joint_prob_list[key] / B_trace_marginal_probs[key.split('_')[1]]) for key in joint_prob_list.keys() if
    #                                            joint_prob_list[key] != 0 and B_trace_marginal_probs[key.split('_')[1]] != 0]).sum()
    #
    #     # the mutual information between A and B trace for this variable thinking that marginal A and B came from each trace distribution
    #     mutual_info_trace = round(np.array(
    #         [joint_prob_list[key] * (np.log(joint_prob_list[key] / (A_trace_marginal_probs[key.split('_')[0]] * B_trace_marginal_probs[key.split('_')[1]])) / np.log(base_of_log)) for key in
    #          joint_prob_list.keys() if joint_prob_list[key] != 0 and B_trace_marginal_probs[key.split('_')[1]] != 0 and A_trace_marginal_probs[key.split('_')[0]] != 0]).sum(), 14)
    #
    #     # round all the probabilities with saferound, guaranteeing that they will add up to 1 at the end, like probabilities should
    #     #
    #     # the way it does this is by the 'difference' strategy:
    #     #
    #     # 'difference' seeks to minimize the sum of the array of the
    #     # differences between the original value and the rounded value of
    #     # each item in the iterable. It will adjust the items with the
    #     # largest difference to preserve the sum. This is the default.
    #     rounded = iteround.saferound(joint_prob_list.values(), precision)
    #     for key, index in zip(joint_prob_list.keys(), range(len(joint_prob_list))):
    #         joint_prob_list[key] = rounded[index]
    #     rounded = iteround.saferound(A_trace_marginal_probs.values(), precision)
    #     for key, index in zip(A_trace_marginal_probs.keys(), range(len(A_trace_marginal_probs))):
    #         A_trace_marginal_probs[key] = rounded[index]
    #     rounded = iteround.saferound(B_trace_marginal_probs.values(), precision)
    #     for key, index in zip(B_trace_marginal_probs.keys(), range(len(B_trace_marginal_probs))):
    #         B_trace_marginal_probs[key] = rounded[index]
    #
    #     # checking joint prob adds up to one, NOTE WE ARE USING INTEGERS BECAUSE OF FLOATING POINT PRECISION ERRORS
    #     if int(np.array([val*(10**precision) for val in joint_prob_list.values()]).sum()) != (10**precision):
    #         print((10**precision), type(10**precision), np.array([int(val*1000) for val in joint_prob_list.values()]).sum(), type(np.array([int(val*1000) for val in joint_prob_list.values()]).sum()))
    #         print('joint prob does not add up to 1.0! it adds up to {}'.format(np.array([val for val in joint_prob_list.values()]).sum()))
    #         print(np.array([int(val*1000) for val in joint_prob_list.values()]))
    #         print(np.array([int(val * 1000) for val in joint_prob_list.values()]).sum())
    #         num = 0
    #         for va in np.array([val for val in joint_prob_list.values()]):
    #             num += va
    #             print(va, type(va))
    #         print(num)
    #         exit()
    #
    #     # checking A marginal prob adds up to one
    #     if int(np.array([val*(10**precision) for val in A_trace_marginal_probs.values()]).sum()) != (10**precision):
    #         print(int(np.array([int(val*(10**precision)) for val in A_trace_marginal_probs.values()]).sum()), (10**precision))
    #         print(np.array([int(val*(10**precision)) for val in A_trace_marginal_probs.values()]))
    #         print(np.array([val * (10 ** precision) for val in A_trace_marginal_probs.values()]).sum())
    #         print(int(np.array([val*(10**precision) for val in A_trace_marginal_probs.values()]).sum()))
    #         print('A_trace_marginal_probs does not add up to 1.0! it adds up to {}'.format(np.array([val for val in A_trace_marginal_probs.values()]).sum()))
    #         exit()
    #
    #     # checking B marginal prob adds up to one
    #     if int(np.array([val*(10**precision) for val in B_trace_marginal_probs.values()]).sum()) != (10**precision):
    #         print('B_trace_marginal_probs does not add up to 1.0! it adds up to {}'.format(np.array([val for val in B_trace_marginal_probs.values()]).sum()))
    #         exit()
    #
    #     # mutual information cannot be negative
    #     if mutual_info_trace < 0:
    #         print('mutual info is negative! something is wrong...')
    #         print(A_cut)
    #         print(B_cut)
    #         print(joint_prob_list)
    #         print(A_trace_marginal_probs)
    #         print(B_trace_marginal_probs)
    #         print(mutual_info_trace)
    #         for key in joint_prob_list.keys():
    #             print('key:', key)
    #             if joint_prob_list[key] != 0 and B_trace_marginal_probs[key.split('_')[1]] != 0 and A_trace_marginal_probs[key.split('_')[0]] != 0:
    #                 print(joint_prob_list[key])
    #                 print(A_trace_marginal_probs[key.split('_')[0]])
    #                 print(B_trace_marginal_probs[key.split('_')[1]])
    #                 print(joint_prob_list[key] * np.log(joint_prob_list[key] / (A_trace_marginal_probs[key.split('_')[0]] * B_trace_marginal_probs[key.split('_')[1]])))
    #         print('_________')
    #         exit()
    #
    #     return mutual_info_trace, A_conditioned_on_B_entropy, B_conditioned_on_A_entropy, joint_prob_list
    #
    # #
    # def get_MI_with_dict(self, A_new, B_new, variable_names, A_variable_symbol, B_variable_symbol, bins_on_side, base_of_log, precision, dictionary, key_format):
    #     for var1 in variable_names:
    #         for var2 in variable_names:
    #             dictionary[key_format.format(A_variable_symbol[var1], B_variable_symbol[var2])].append(
    #                 self.get_entropies_and_joint_probs_from_labeled_data(A_cut=A_new[var1], B_cut=B_new[var2], bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision)[0])
    #
    #     return dictionary
    #
    # #
    # def get_MI_df(self, A_categories, B_categories, variable_names, A_variable_symbol, B_variable_symbol, bins_on_side, base_of_log, precision):
    #     dataset_MI_mean = pd.DataFrame(index=['B_' + var for var in variable_names], columns=['A_' + var for var in variable_names], dtype=float)
    #     for var1 in variable_names:
    #         for var2 in variable_names:
    #             dataset_MI_mean['A_' + var1].loc['B_' + var2] = \
    #             self.get_entropies_and_joint_probs_from_labeled_data(A_cut=A_categories[var1], B_cut=B_categories[var2], bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision)[0]
    #
    #     dataset_MI_mean = dataset_MI_mean.rename(columns=dict(zip(dataset_MI_mean.columns, B_variable_symbol)), index=dict(zip(dataset_MI_mean.index, A_variable_symbol)))
    #
    #     return dataset_MI_mean
    #
    # """ do the heatmaps for the three MI datasets """
    #
    # def calculate_MI_and_save_heatmap_for_all_dsets_together(self, sis_A_pooled, sis_B_pooled, non_sis_A_pooled, non_sis_B_pooled, con_A_pooled, con_B_pooled, bins_on_side, type_mean, variable_names,
    #         A_variable_symbol, B_variable_symbol, mult_number, directory, base_of_log, precision):
    #
    #
    #     sis_MI = self.get_MI_df(sis_A_pooled, sis_B_pooled, variable_names, A_variable_symbol, B_variable_symbol, bins_on_side, base_of_log, precision) * mult_number
    #     non_sis_MI = self.get_MI_df(non_sis_A_pooled, non_sis_B_pooled, variable_names, A_variable_symbol, B_variable_symbol, bins_on_side, base_of_log, precision) * mult_number
    #     con_MI = self.get_MI_df(con_A_pooled, con_B_pooled, variable_names, A_variable_symbol, B_variable_symbol, bins_on_side, base_of_log, precision) * mult_number
    #
    #     vmin = np.min(np.array([np.min(sis_MI.min()), np.min(non_sis_MI.min()), np.min(con_MI.min())]))
    #     vmax = np.max(np.array([np.max(sis_MI.max()), np.max(non_sis_MI.max()), np.max(con_MI.max())]))
    #
    #     fig, (ax_sis, ax_non_sis, ax_con) = plt.subplots(ncols=3, figsize=(12.7, 7.5))
    #     fig.subplots_adjust(wspace=0.01)
    #
    #     sns.heatmap(data=sis_MI, annot=True, ax=ax_sis, cbar=False, vmin=vmin, vmax=vmax, yticklabels=True, fmt='.0f')
    #     ax_sis.set_title('Sister')
    #     sns.heatmap(data=non_sis_MI, annot=True, ax=ax_non_sis, cbar=False, vmin=vmin, vmax=vmax, yticklabels=False, fmt='.0f')
    #     ax_non_sis.set_title('Non-Sister')
    #     sns.heatmap(data=con_MI, annot=True, ax=ax_con, cbar=True, vmin=vmin, vmax=vmax, yticklabels=False, fmt='.0f')  # xticklabels=[]
    #     ax_con.set_title('Control')
    #     fig.suptitle('Mutual Information with {} bins on each side, multiplied by {}'.format(bins_on_side, mult_number))
    #     plt.tight_layout(rect=(0, 0, 1, .96))
    #     # fig.colorbar(ax_con.collections[0], ax=ax_con, location="right", use_gridspec=False, pad=0.2)
    #     # plt.title('{} Mutual Information with {} bins on each side of the {} mean'.format(dataset, bins_on_side, type_mean))
    #     # plt.show()
    #     plt.savefig(directory + '/Mutual Information with {} bins on each side, multiplied by {}'.format(bins_on_side, mult_number), dpi=300)
    #     plt.close()
    #
    # """ do the heatmap for the one MI dataset (Inter/Same-Cell) """
    #
    # def calculate_MI_and_save_heatmap(self, A_categories, B_categories, bins_on_side, base_of_log, precision, dataset, type_mean, variable_names, A_variable_symbol, B_variable_symbol, mult_number, directory, **kwargs):
    #
    #     half_matrix = kwargs.get('half_matrix', False)
    #
    #     dataset_MI_mean = self.get_MI_df(A_categories=A_categories, B_categories=B_categories, variable_names=variable_names, A_variable_symbol=A_variable_symbol, B_variable_symbol=B_variable_symbol,
    #         bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision) * mult_number
    #
    #     plt.figure(figsize=(12.7, 7.5))
    #     if half_matrix == True:
    #         # so that the color scale corresponds to the subdiagonal heatmap and not the whole heatmap
    #
    #         # This is to plot only the subdiagonal heatmap
    #         mask = np.zeros_like(dataset_MI_mean)
    #         mask[np.triu_indices_from(mask)] = True
    #
    #         vals_array = []
    #         for ind1 in range(1, len(A_variable_symbol)):
    #             for ind2 in range(ind1):
    #                 vals_array.append(dataset_MI_mean[A_variable_symbol[ind1]].loc[A_variable_symbol[ind2]])
    #         vals_array = np.array(vals_array)
    #
    #         vmax = np.max(vals_array)
    #         vmin = np.min(vals_array)
    #
    #         sns.heatmap(data=dataset_MI_mean, annot=True, fmt='.0f', mask=mask, vmax=vmax, vmin=vmin)
    #     else:
    #         sns.heatmap(data=dataset_MI_mean, annot=True, fmt='.0f')
    #     plt.title('{}, {} bins on each side of the {}, multiplied by {}'.format(dataset, bins_on_side, type_mean, mult_number))
    #     plt.show()
    #     # plt.savefig(directory + '/{}, {} bins on each side of the {}'.format(dataset, bins_on_side, type_mean), dpi=300)
    #     plt.close()
    #
    # """ get the confidence intervals via bootstrapping """
    #
    # def bootstrap_bias_corrected_accelerated_confidence_intervals(self, A_df, B_df, variable_names, how_many_times_to_resample, percent_to_cover, calculate_acceleration, bins_on_side, base_of_log, precision, show_histograms, **kwargs):
    #
    #     def get_MI_dict(A_new, B_new, variable_names, A_variable_symbol, B_variable_symbol, bins_on_side, base_of_log, precision, dictionary, key_format):
    #         for var1 in variable_names:
    #             for var2 in variable_names:
    #                 dictionary[key_format.format(A_variable_symbol[var1], B_variable_symbol[var2])].append(
    #                     self.get_entropies_and_joint_probs_from_labeled_data(A_cut=A_new[var1], B_cut=B_new[var2], bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision)[0])
    #
    #         return dictionary
    #
    #     def get_MI_estimate_dict(A_new, B_new, variable_names, A_variable_symbol, B_variable_symbol, bins_on_side, base_of_log, precision, dictionary, key_format, knns):
    #         for var1 in variable_names:
    #             for var2 in variable_names:
    #                 dictionary[key_format.format(A_variable_symbol[var1], B_variable_symbol[var2])].append(
    #                     mutual_info_regression(X=np.array(A_new[var1]).reshape(-1, 1), y=np.array(B_new[var2]), discrete_features=False, n_neighbors=knns, copy=True, random_state=42)[0])
    #
    #         return dictionary
    #
    #     knns = kwargs.get('knns', False) # If any other number rather than zero, we use the estimation method to get the Mutual Information and knns as the k nearest neighbors
    #
    #     # warnings
    #     if (percent_to_cover >= 1) or (percent_to_cover <= 0):
    #         raise IOError('percent_to_cover must be strictly between 0 and 1')
    #
    #     # get the MIs with the original dataset
    #     original_dictionary_of_MIs = {r'$I({};{})$'.format(A_var, B_var): [] for A_var in self.A_variable_symbols.values() for B_var in self.B_variable_symbols.values()}
    #     if knns:
    #         original_dictionary_of_MIs = get_MI_estimate_dict(A_df, B_df, variable_names, self.A_variable_symbols, self.B_variable_symbols, 1, 2, 2, original_dictionary_of_MIs, r'$I({};{})$', knns)
    #     else:
    #         original_dictionary_of_MIs = get_MI_dict(A_df, B_df, variable_names, self.A_variable_symbols, self.B_variable_symbols, 1, 2, 2, original_dictionary_of_MIs, r'$I({};{})$')
    #
    #     # for plotting purposes
    #     A_variable_symbol = dict(zip(variable_names, [r'\ln(\tau)_A', r'\ln(x(0))_A', r'\ln(x(\tau))_A', r'\ln(\alpha)_A', r'\ln(\phi)_A', r'f_A', r'\Delta_A']))
    #     B_variable_symbol = dict(zip(variable_names, [r'\ln(\tau)_B', r'\ln(x(0))_B', r'\ln(x(\tau))_B', r'\ln(\alpha)_B', r'\ln(\phi)_B', r'f_B', r'\Delta_B']))
    #
    #     # to put arrays of length how_many_times_to_resample that are the bootstrapped Mutual Information
    #     bootstrap_dicitonary_of_MIs = {r'$I({};{})$'.format(A_var, B_var): [] for A_var in A_variable_symbol.values() for B_var in B_variable_symbol.values()}
    #
    #     # decide if we want to calculate the acceleration or if it is already calculated
    #     if calculate_acceleration:
    #         print('calculating acceleration')
    #         acceleration = dict()
    #         # calculate the acceleration, we don't need the bootstrap
    #         for A_var in variable_names:
    #             for B_var in variable_names:
    #                 if knns:
    #                     dropped_index_MIs = np.array([
    #                         mutual_info_regression(X=np.array(A_df[A_var].drop(index=index)).reshape(-1, 1), y=np.array(B_df[B_var].drop(index=index)), discrete_features=False, n_neighbors=knns,
    #                             copy=True, random_state=42)[0] for index in np.arange(len(A_df))])
    #                 else:
    #                     dropped_index_MIs = np.array([
    #                         self.get_entropies_and_joint_probs_from_labeled_data(A_cut=A_df[A_var].drop(index=index), B_cut=B_df[B_var].drop(index=index), bins_on_side=bins_on_side,
    #                             base_of_log=base_of_log, precision=precision)[0] for index in np.arange(len(A_df))])
    #                 constant = dropped_index_MIs.sum()
    #                 a_numerator = np.array([(constant - dropped_MI) ** 3 for dropped_MI in dropped_index_MIs]).sum()
    #                 a_denominator = 6 * (np.array([(constant - dropped_MI) ** 2 for dropped_MI in dropped_index_MIs]).sum() ** (1.5))
    #                 a = a_numerator / a_denominator
    #                 acceleration.update({r'$I({};{})$'.format(A_var, B_var): a})
    #                 print('-------')
    #                 print(r'$I({};{})$'.format(A_variable_symbol[A_var], B_variable_symbol[B_var]))
    #                 print('len(dropped_index_MIs):', len(dropped_index_MIs))
    #                 print('constant:', constant)
    #                 print('a_numerator:', a_numerator)
    #                 print('a_denominator:', a_denominator)
    #                 print('a:', a)
    #         exit()
    #     else:
    #         acceleration = {key: 0.00284123 for key in bootstrap_dicitonary_of_MIs.keys()} # gotten from previously calculating them all
    #
    #     # bootstrap and calculate the Mutual Information on this resampled dataset with replacement
    #     for resample_number in range(how_many_times_to_resample):
    #         print('resample number:', resample_number)
    #         A_new = A_df.sample(n=len(A_df), replace=True, axis='index')
    #         B_new = B_df.loc[A_new.index]
    #
    #         if knns:
    #             bootstrap_dicitonary_of_MIs = get_MI_estimate_dict(A_new, B_new, variable_names, A_variable_symbol, B_variable_symbol, bins_on_side, base_of_log, precision, bootstrap_dicitonary_of_MIs,
    #                 r'$I({};{})$', knns)
    #         else:
    #             bootstrap_dicitonary_of_MIs = get_MI_dict(A_new, B_new, variable_names, A_variable_symbol, B_variable_symbol, bins_on_side, base_of_log, precision, bootstrap_dicitonary_of_MIs, r'$I({};{})$')
    #
    #     # add the original MI to the bootsrap array
    #     for key in bootstrap_dicitonary_of_MIs.keys():
    #         bootstrap_dicitonary_of_MIs[key].append(original_dictionary_of_MIs[key][0])
    #
    #     # acceleration = 0.00284123 # gotten from previous calculations
    #     BC_a_confidence_interval = dict()
    #
    #     # for the percentile points
    #     alpha = (1.0 - percent_to_cover) / 2.0
    #
    #     # caculate the confidence intervals based on the bias and acceleration
    #     for A_var in variable_names:
    #         for B_var in variable_names:
    #             key = r'$I({};{})$'.format(A_variable_symbol[A_var], B_variable_symbol[B_var])
    #
    #             # print('How many were lower or as low as original', [True for val in bootstrap_dicitonary_of_MIs[key] if val <= original_dictionary_of_MIs[key]])
    #             # calculating z naught which measures discrepancy of median versus the estimate
    #             bias_z_naught = stats.norm.ppf(len([True for val in bootstrap_dicitonary_of_MIs[key] if val <= original_dictionary_of_MIs[key]]) / how_many_times_to_resample)
    #
    #             # print(key, len([True for val in bootstrap_dicitonary_of_MIs[key] if val <
    #             #                 self.get_entropies_and_joint_probs_from_labeled_data(A_cut=A_df[A_var], B_cut=B_df[B_var], bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision)[
    #             #                     0]]) / how_many_times_to_resample)
    #
    #             # get the lower and upper bounds
    #             # print(key, bias_z_naught + (bias_z_naught + stats.norm.ppf(alpha))/(1 - acceleration[key] * (bias_z_naught + stats.norm.ppf(alpha))))
    #             # print('stats.norm.ppf(alpha)', stats.norm.ppf(alpha))
    #             # print('bias_z_naught', bias_z_naught)
    #             # print('bias_z_naught + (bias_z_naught + stats.norm.ppf(alpha))/(1 - acceleration[key] * (bias_z_naught + stats.norm.ppf(alpha)))', bias_z_naught + (bias_z_naught + stats.norm.ppf(alpha))/(1 - acceleration[key] * (bias_z_naught + stats.norm.ppf(alpha))))
    #             alpha_lower = stats.norm.cdf(bias_z_naught + (bias_z_naught + stats.norm.ppf(alpha))/(1 - acceleration[key] * (bias_z_naught + stats.norm.ppf(alpha))))
    #             # print('alpha_lower', alpha_lower)
    #             alpha_upper = stats.norm.cdf(bias_z_naught + (bias_z_naught + stats.norm.ppf(1 - alpha)) / (1 - acceleration[key] * (bias_z_naught + stats.norm.ppf(1 - alpha))))
    #             # print('alpha_upper', alpha_upper)
    #             lower_bound = np.percentile(np.sort(bootstrap_dicitonary_of_MIs[r'$I({};{})$'.format(A_variable_symbol[A_var], B_variable_symbol[B_var])]), 100 * alpha_lower)
    #             upper_bound = np.percentile(np.sort(bootstrap_dicitonary_of_MIs[r'$I({};{})$'.format(A_variable_symbol[A_var], B_variable_symbol[B_var])]), 100 * alpha_upper)
    #
    #             # update the confidence interval dictionary
    #             BC_a_confidence_interval.update({r'$I({};{})$'.format(A_variable_symbol[A_var], B_variable_symbol[B_var]): [lower_bound, upper_bound]})
    #
    #     # show the different types of confidence intervals to compare
    #     if show_histograms == True:
    #         for key, val in bootstrap_dicitonary_of_MIs.items():
    #             # print(key, np.array(val), type(val[0]))
    #             sns.distplot(val)
    #             # sns.set_style('whitegrid')
    #             # plt.hist(val, density=True, cumulative=True, bins=int(np.ceil(np.sqrt(len(val)))))
    #             sorted = np.sort(val)
    #             left_bndry = np.percentile(val, 100*alpha)
    #             right_bndry = np.percentile(val, 100*(1.0 - alpha))
    #             plt.title(key)
    #             print('---------')
    #             print(left_bndry, right_bndry, np.median(sorted))
    #             print(BC_a_confidence_interval[key][0], BC_a_confidence_interval[key][1], np.median(sorted))
    #             print('Percentile {}% of the sample inside'.format(round(len([num for num in val if (num <= right_bndry) and (num >= left_bndry)]) / how_many_times_to_resample, 2)))
    #             print('Bias corrected, {}% of the sample inside'.format(round(len([num for num in val if (num <= BC_a_confidence_interval[key][1]) and (num >= BC_a_confidence_interval[key][0])]) / how_many_times_to_resample, 2)))
    #             plt.axvline(left_bndry, color='black')
    #             plt.axvline(right_bndry, color='black')
    #             plt.axvline(BC_a_confidence_interval[key][0], color='green')
    #             plt.axvline(BC_a_confidence_interval[key][1], color='green')
    #             plt.axvline(np.median(val), color='orange', ls='--', label='Median')
    #             plt.show()
    #             plt.close()
    #
    #     # quantile_boundaries_on_the_MIs = {key: [np.percentile(np.sort(bootstrap_dicitonary_of_MIs[key]), 10), np.percentile(np.sort(bootstrap_dicitonary_of_MIs[key]), 90)] for key in bootstrap_dicitonary_of_MIs.keys()}
    #
    #     return {'original': original_dictionary_of_MIs, 'bootstrap': bootstrap_dicitonary_of_MIs, 'confidence_intervals': BC_a_confidence_interval}




    """ outputs the regression matrix with the coefficients, the scores, and the intercepts of all regression of target variables from the target 
        dataframe using the factor variables from the factor dataframe """

    def linear_regression_framework(self, df_of_avgs, factor_variables, target_variables, factor_df, target_df, fit_intercept):

        # The centered variables, we are trying to find how much they depend on what centers them
        centered_fv = ['centered_' + fv for fv in factor_variables]
        centered_tv = ['centered_' + tv for tv in target_variables]

        # all the data we need to make the prediction
        factors = factor_df[factor_variables].rename(columns=dict(zip(factor_variables, centered_fv))) - df_of_avgs.rename(columns=dict(zip(list(df_of_avgs.columns), centered_fv)))

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
                plt.scatter(self._data[dataID]['time' + ks].iloc[start_indices], self._data[dataID][self._discretization_variable + ks].iloc[start_indices], color='red', label='first point in cycle')
                plt.scatter(self._data[dataID]['time' + ks].iloc[end_indices], self._data[dataID][self._discretization_variable + ks].iloc[end_indices], color='green', label='last point in cycle')
                plt.legend()

                # save them or show them based on what we answered
                if answer == 'save':
                    plt.savefig(name + '/' + ks + '_' + str(dataID) + '.png', dpi=300)
                elif answer == 'see':
                    plt.show()
                else:
                    plt.show()
                plt.close()

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

    def plot_relationship_correlations(self, df_1, df_2, df_1_variables, df_2_variables, x_labels, y_labels, limit_the_axes_func=limit_the_axes, corr_to_annotate_func=corrfunc):
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
            stats[variable].loc['std'] = np.std(np.concatenate([_all_data_dict[dataID][variable] for dataID in _all_data_dict.keys()]), ddof=1)

        return stats

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
            
    # """ for the bins_on_side method """
    #
    # def get_the_category_and_joint_prob_for_datasets(self, **kwargs):
    #
    #     precision = kwargs.get('precision', 2) # for the joint distributions to add up to 1 and this sets how many sig-digs are shown in the heatmap
    #     bins_on_side = kwargs.get('bins_on_side', 1) # for this bins_on_side method
    #     base_of_log = 2 * bins_on_side
    #
    #     # the dictionary of categories
    #     self.category_and_joint_prob_per_dataset_dictionary = dict()
    #
    #     if 'Sisters' in self.datasets:
    #         # get the bins_on_side_categories
    #         sis_A_bins_on_side_categories, sis_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.sis_A_pooled, B_df=self.sis_B_pooled,
    #             log_A_df=self.sis_log_A_pooled, log_B_df=self.sis_log_B_pooled, bins_on_side=bins_on_side, variable_names=self._variable_names,
    #             log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #         # print the joint distributions heatmaps
    #         joint_probs_dict = self.get_joint_probs_dict(sis_A_bins_on_side_categories, sis_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision)
    #
    #         # update the dictionary
    #         self.category_and_joint_prob_per_dataset_dictionary.update({'sis': {'categories_df': [sis_A_bins_on_side_categories, sis_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #         # GET THE INTRA GENERATIONS
    #         for index in range(len(self.sis_A_intra_gen_bacteria)):
    #             # get the bins_on_side_categories
    #             sis_intra_A_bins_on_side_categories, sis_intra_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.sis_A_intra_gen_bacteria[index], B_df=self.sis_B_intra_gen_bacteria[index],
    #                 log_A_df=self.sis_log_A_intra_gen_bacteria[index], log_B_df=self.sis_log_B_intra_gen_bacteria[index], bins_on_side=bins_on_side, variable_names=self._variable_names,
    #                 log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #             # print the joint distributions heatmaps
    #             joint_probs_dict = self.get_joint_probs_dict(sis_intra_A_bins_on_side_categories, sis_intra_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side,
    #                 base_of_log=base_of_log, precision=precision)
    #
    #             # update the dictionary
    #             self.category_and_joint_prob_per_dataset_dictionary.update(
    #                 {'sis_intra_{}'.format(index): {'categories_df': [sis_intra_A_bins_on_side_categories, sis_intra_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #     if 'Nonsisters' in self.datasets:
    #         # get the bins_on_side_categories
    #         non_sis_A_bins_on_side_categories, non_sis_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.non_sis_A_pooled, B_df=self.non_sis_B_pooled,
    #             log_A_df=self.non_sis_log_A_pooled, log_B_df=self.non_sis_log_B_pooled, bins_on_side=bins_on_side, variable_names=self._variable_names,
    #             log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #         # print the joint distributions heatmaps
    #         joint_probs_dict = self.get_joint_probs_dict(non_sis_A_bins_on_side_categories, non_sis_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side,
    #             base_of_log=base_of_log, precision=precision)
    #
    #         # update the dictionary
    #         self.category_and_joint_prob_per_dataset_dictionary.update({'non_sis': {'categories_df': [non_sis_A_bins_on_side_categories, non_sis_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #         # GET THE INTRA GENERATIONS
    #         for index in range(len(self.non_sis_A_intra_gen_bacteria)):
    #             # get the bins_on_side_categories
    #             non_sis_intra_A_bins_on_side_categories, non_sis_intra_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.non_sis_A_intra_gen_bacteria[index],
    #                 B_df=self.non_sis_B_intra_gen_bacteria[index], log_A_df=self.non_sis_log_A_intra_gen_bacteria[index], log_B_df=self.non_sis_log_B_intra_gen_bacteria[index], bins_on_side=bins_on_side,
    #                 variable_names=self._variable_names, log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #             # print the joint distributions heatmaps
    #             joint_probs_dict = self.get_joint_probs_dict(non_sis_intra_A_bins_on_side_categories, non_sis_intra_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side,
    #                 base_of_log=base_of_log, precision=precision)
    #
    #             # update the dictionary
    #             self.category_and_joint_prob_per_dataset_dictionary.update(
    #                 {'non_sis_intra_{}'.format(index): {'categories_df': [non_sis_intra_A_bins_on_side_categories, non_sis_intra_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #     if 'Control' in self.datasets:
    #         # get the bins_on_side_categories
    #         con_A_bins_on_side_categories, con_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.con_A_pooled, B_df=self.con_B_pooled,
    #             log_A_df=self.con_log_A_pooled, log_B_df=self.con_log_B_pooled, bins_on_side=bins_on_side, variable_names=self._variable_names,
    #             log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #         # print the joint distributions heatmaps
    #         joint_probs_dict = self.get_joint_probs_dict(con_A_bins_on_side_categories, con_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision)
    #
    #         # update the dictionary
    #         self.category_and_joint_prob_per_dataset_dictionary.update({'con': {'categories_df': [con_A_bins_on_side_categories, con_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #         # GET THE INTRA GENERATIONS
    #         for index in range(len(self.con_A_intra_gen_bacteria)):
    #             # get the bins_on_side_categories
    #             con_intra_A_bins_on_side_categories, con_intra_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.con_A_intra_gen_bacteria[index],
    #                 B_df=self.con_B_intra_gen_bacteria[index], log_A_df=self.con_log_A_intra_gen_bacteria[index], log_B_df=self.con_log_B_intra_gen_bacteria[index], bins_on_side=bins_on_side,
    #                 variable_names=self._variable_names, log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #             # print the joint distributions heatmaps
    #             joint_probs_dict = self.get_joint_probs_dict(con_intra_A_bins_on_side_categories, con_intra_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side,
    #                 base_of_log=base_of_log, precision=precision)
    #
    #             # update the dictionary
    #             self.category_and_joint_prob_per_dataset_dictionary.update(
    #                 {'con_intra_{}'.format(index): {'categories_df': [con_intra_A_bins_on_side_categories, con_intra_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #     if 'Population' in self.datasets:
    #
    #         for index in range(len(self.log_daughter_dfs)):
    #             # do for mother-daughter
    #             inter_mom_categories, inter_daughter_categories = self.put_the_bins_on_side_categories(A_df=self.mother_dfs[index], B_df=self.daughter_dfs[index], log_A_df=self.log_mother_dfs[index],
    #                 log_B_df=self.log_daughter_dfs[index], bins_on_side=bins_on_side, variable_names=self._variable_names,
    #                 log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #             # print the joint distributions heatmaps
    #             joint_probs_dict = self.get_joint_probs_dict(inter_mom_categories, inter_daughter_categories, self._variable_names, bins_on_side=bins_on_side, base_of_log=base_of_log,
    #                 precision=precision)
    #
    #             # update the dictionary
    #             self.category_and_joint_prob_per_dataset_dictionary.update({'inter_{}'.format(index): {'categories_df': [inter_mom_categories, inter_daughter_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #         # do for same-cell
    #         all_bacteria_categories_0, all_bacteria_categories_1 = self.put_the_bins_on_side_categories(A_df=self.all_bacteria, B_df=self.all_bacteria, log_A_df=self.log_all_bacteria,
    #             log_B_df=self.log_all_bacteria, bins_on_side=bins_on_side, variable_names=self._variable_names,
    #             log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #         # print the joint distributions heatmaps
    #         joint_probs_dict = self.get_joint_probs_dict(all_bacteria_categories_0, all_bacteria_categories_1, self._variable_names, bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision)
    #
    #         # update the dictionary
    #         self.category_and_joint_prob_per_dataset_dictionary.update({'same_cell': {'categories_df': [all_bacteria_categories_0, all_bacteria_categories_1], 'joint_probs_dict': joint_probs_dict}})
    #
    # """ for the bins_on_side method """
    #
    # def get_the_category_and_joint_prob_for_measurements(self, **kwargs):
    #
    #     precision = kwargs.get('precision', 2)  # for the joint distributions to add up to 1 and this sets how many sig-digs are shown in the heatmap
    #     bins_on_side = kwargs.get('bins_on_side', 1)  # for this bins_on_side method
    #     base_of_log = 2 * bins_on_side
    #
    #     # the dictionary of categories
    #     self.category_and_joint_prob_per_dataset_dictionary = dict()
    #
    #     if 'Sisters' in self.datasets:
    #         # get the bins_on_side_categories
    #         sis_A_bins_on_side_categories, sis_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.sis_A_pooled, B_df=self.sis_B_pooled, log_A_df=self.sis_log_A_pooled,
    #             log_B_df=self.sis_log_B_pooled, bins_on_side=bins_on_side, variable_names=self._variable_names,
    #             log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #         # print the joint distributions heatmaps
    #         joint_probs_dict = self.get_joint_probs_dict(sis_A_bins_on_side_categories, sis_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side, base_of_log=base_of_log,
    #             precision=precision)
    #
    #         # update the dictionary
    #         self.category_and_joint_prob_per_dataset_dictionary.update({'sis': {'categories_df': [sis_A_bins_on_side_categories, sis_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #         # GET THE INTRA GENERATIONS
    #         for index in range(len(self.sis_A_intra_gen_bacteria)):
    #             # get the bins_on_side_categories
    #             sis_intra_A_bins_on_side_categories, sis_intra_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.sis_A_intra_gen_bacteria[index],
    #                 B_df=self.sis_B_intra_gen_bacteria[index], log_A_df=self.sis_log_A_intra_gen_bacteria[index], log_B_df=self.sis_log_B_intra_gen_bacteria[index], bins_on_side=bins_on_side,
    #                 variable_names=self._variable_names, log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #             # print the joint distributions heatmaps
    #             joint_probs_dict = self.get_joint_probs_dict(sis_intra_A_bins_on_side_categories, sis_intra_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side,
    #                 base_of_log=base_of_log, precision=precision)
    #
    #             # update the dictionary
    #             self.category_and_joint_prob_per_dataset_dictionary.update(
    #                 {'sis_intra_{}'.format(index): {'categories_df': [sis_intra_A_bins_on_side_categories, sis_intra_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #     if 'Nonsisters' in self.datasets:
    #         # get the bins_on_side_categories
    #         non_sis_A_bins_on_side_categories, non_sis_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.non_sis_A_pooled, B_df=self.non_sis_B_pooled,
    #             log_A_df=self.non_sis_log_A_pooled, log_B_df=self.non_sis_log_B_pooled, bins_on_side=bins_on_side, variable_names=self._variable_names,
    #             log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #         # print the joint distributions heatmaps
    #         joint_probs_dict = self.get_joint_probs_dict(non_sis_A_bins_on_side_categories, non_sis_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side, base_of_log=base_of_log,
    #             precision=precision)
    #
    #         # update the dictionary
    #         self.category_and_joint_prob_per_dataset_dictionary.update(
    #             {'non_sis': {'categories_df': [non_sis_A_bins_on_side_categories, non_sis_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #         # GET THE INTRA GENERATIONS
    #         for index in range(len(self.non_sis_A_intra_gen_bacteria)):
    #             # get the bins_on_side_categories
    #             non_sis_intra_A_bins_on_side_categories, non_sis_intra_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.non_sis_A_intra_gen_bacteria[index],
    #                 B_df=self.non_sis_B_intra_gen_bacteria[index], log_A_df=self.non_sis_log_A_intra_gen_bacteria[index], log_B_df=self.non_sis_log_B_intra_gen_bacteria[index],
    #                 bins_on_side=bins_on_side, variable_names=self._variable_names, log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #             # print the joint distributions heatmaps
    #             joint_probs_dict = self.get_joint_probs_dict(non_sis_intra_A_bins_on_side_categories, non_sis_intra_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side,
    #                 base_of_log=base_of_log, precision=precision)
    #
    #             # update the dictionary
    #             self.category_and_joint_prob_per_dataset_dictionary.update(
    #                 {'non_sis_intra_{}'.format(index): {'categories_df': [non_sis_intra_A_bins_on_side_categories, non_sis_intra_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #     if 'Control' in self.datasets:
    #         # get the bins_on_side_categories
    #         con_A_bins_on_side_categories, con_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.con_A_pooled, B_df=self.con_B_pooled, log_A_df=self.con_log_A_pooled,
    #             log_B_df=self.con_log_B_pooled, bins_on_side=bins_on_side, variable_names=self._variable_names,
    #             log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #         # print the joint distributions heatmaps
    #         joint_probs_dict = self.get_joint_probs_dict(con_A_bins_on_side_categories, con_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side, base_of_log=base_of_log,
    #             precision=precision)
    #
    #         # update the dictionary
    #         self.category_and_joint_prob_per_dataset_dictionary.update({'con': {'categories_df': [con_A_bins_on_side_categories, con_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #         # GET THE INTRA GENERATIONS
    #         for index in range(len(self.con_A_intra_gen_bacteria)):
    #             # get the bins_on_side_categories
    #             con_intra_A_bins_on_side_categories, con_intra_B_bins_on_side_categories = self.put_the_bins_on_side_categories(A_df=self.con_A_intra_gen_bacteria[index],
    #                 B_df=self.con_B_intra_gen_bacteria[index], log_A_df=self.con_log_A_intra_gen_bacteria[index], log_B_df=self.con_log_B_intra_gen_bacteria[index], bins_on_side=bins_on_side,
    #                 variable_names=self._variable_names, log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #             # print the joint distributions heatmaps
    #             joint_probs_dict = self.get_joint_probs_dict(con_intra_A_bins_on_side_categories, con_intra_B_bins_on_side_categories, self._variable_names, bins_on_side=bins_on_side,
    #                 base_of_log=base_of_log, precision=precision)
    #
    #             # update the dictionary
    #             self.category_and_joint_prob_per_dataset_dictionary.update(
    #                 {'con_intra_{}'.format(index): {'categories_df': [con_intra_A_bins_on_side_categories, con_intra_B_bins_on_side_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #     if 'Population' in self.datasets:
    #
    #         for index in range(len(self.log_daughter_dfs)):
    #             # do for mother-daughter
    #             inter_mom_categories, inter_daughter_categories = self.put_the_bins_on_side_categories(A_df=self.mother_dfs[index], B_df=self.daughter_dfs[index], log_A_df=self.log_mother_dfs[index],
    #                 log_B_df=self.log_daughter_dfs[index], bins_on_side=bins_on_side, variable_names=self._variable_names,
    #                 log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #             # print the joint distributions heatmaps
    #             joint_probs_dict = self.get_joint_probs_dict(inter_mom_categories, inter_daughter_categories, self._variable_names, bins_on_side=bins_on_side, base_of_log=base_of_log,
    #                 precision=precision)
    #
    #             # update the dictionary
    #             self.category_and_joint_prob_per_dataset_dictionary.update(
    #                 {'inter_{}'.format(index): {'categories_df': [inter_mom_categories, inter_daughter_categories], 'joint_probs_dict': joint_probs_dict}})
    #
    #         # do for same-cell
    #         all_bacteria_categories_0, all_bacteria_categories_1 = self.put_the_bins_on_side_categories(A_df=self.all_bacteria, B_df=self.all_bacteria, log_A_df=self.log_all_bacteria,
    #             log_B_df=self.log_all_bacteria, bins_on_side=bins_on_side, variable_names=self._variable_names,
    #             log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    #
    #         # print the joint distributions heatmaps
    #         joint_probs_dict = self.get_joint_probs_dict(all_bacteria_categories_0, all_bacteria_categories_1, self._variable_names, bins_on_side=bins_on_side, base_of_log=base_of_log,
    #             precision=precision)
    #
    #         # update the dictionary
    #         self.category_and_joint_prob_per_dataset_dictionary.update({'same_cell': {'categories_df': [all_bacteria_categories_0, all_bacteria_categories_1], 'joint_probs_dict': joint_probs_dict}})
    #
    # """ This is to be used once the data is processed """
    #
    # def script_to_output_mutual_information_heatmaps(self, **kwargs):
    #
    #     # joint_probs = ['Sisters', 'Nonsisters', 'Control', 'Same_cell', 'Inter_0', ..., 'Intra_0_sis/non_sis/con', ...]
    #     # mutual_informations = ['3 datasets', 'Inter_0', 'Inter_1', ..., 'Same_cell', '3 datasets Intra_0', '3 datasets Intra_1', ...]
    #
    #     mult_number = kwargs.get('mult_number', 100000) # to get the numbers neatly on the heatmaps
    #     precision = kwargs.get('precision', 2) # for the joint distributions to add up to 1 and this sets how many sig-digs are shown in the heatmap
    #     bins_on_side = kwargs.get('bins_on_side', 1) # for this bins_on_side method
    #     base_of_log = 2 * bins_on_side
    #
    #     # get the category_and_joint_prob_per_dataset_dictionary
    #     self.get_the_category_and_joint_prob_for_datasets()
    #
    #     # create the folder if not created already
    #     try:
    #         # Create target Directory
    #         os.mkdir(self.what_to_subtract)
    #         print("Directory ", self.what_to_subtract, " Created ")
    #     except FileExistsError:
    #         print("Directory ", self.what_to_subtract, " already exists")
    #
    #     # plot the joint probability for all possible
    #     for key in self.category_and_joint_prob_per_dataset_dictionary.keys():
    #         self.show_the_joint_dists(self.category_and_joint_prob_per_dataset_dictionary[key]['joint_probs_dict'], self._variable_names, self._variable_names, bins_on_side=bins_on_side,
    #         mean=self.what_to_subtract, dataset=key, directory=self.what_to_subtract, precision=precision)
    #
    #     # save the three dataset MI heatmaps
    #     if ['Sisters', 'Nonsisters', 'Control'] in self.datasets:
    #         # The three datasets
    #         self.calculate_MI_and_save_heatmap_for_all_dsets_together(self.category_and_joint_prob_per_dataset_dictionary['sis']['categories_df'][0],
    #             self.category_and_joint_prob_per_dataset_dictionary['sis']['categories_df'][1], self.category_and_joint_prob_per_dataset_dictionary['non_sis']['categories_df'][0],
    #             self.category_and_joint_prob_per_dataset_dictionary['non_sis']['categories_df'][1], self.category_and_joint_prob_per_dataset_dictionary['con']['categories_df'][0],
    #             self.category_and_joint_prob_per_dataset_dictionary['con']['categories_df'][1], bins_on_side=bins_on_side, type_mean=self.what_to_subtract, variable_names=self._variable_names,
    #             A_variable_symbol=self.A_variable_symbols, B_variable_symbol=self.B_variable_symbols, mult_number=mult_number, directory=self.what_to_subtract, base_of_log=base_of_log, precision=precision)
    #
    #         # The three intra datasets
    #         for index in range(7):
    #             self.calculate_MI_and_save_heatmap_for_all_dsets_together(self.category_and_joint_prob_per_dataset_dictionary['sis_intra_{}'.format(index)]['categories_df'][0],
    #                 self.category_and_joint_prob_per_dataset_dictionary['sis_intra_{}'.format(index)]['categories_df'][1],
    #                 self.category_and_joint_prob_per_dataset_dictionary['non_sis_intra_{}'.format(index)]['categories_df'][0],
    #                 self.category_and_joint_prob_per_dataset_dictionary['non_sis_intra_{}'.format(index)]['categories_df'][1],
    #                 self.category_and_joint_prob_per_dataset_dictionary['con_intra_{}'.format(index)]['categories_df'][0],
    #                 self.category_and_joint_prob_per_dataset_dictionary['con_intra_{}'.format(index)]['categories_df'][1], bins_on_side=bins_on_side, type_mean=self.what_to_subtract,
    #                 variable_names=self._variable_names, A_variable_symbol=self.A_variable_symbols, B_variable_symbol=self.B_variable_symbols, mult_number=mult_number, directory=self.what_to_subtract,
    #                 base_of_log=base_of_log, precision=precision)
    #
    #     # for the inter and same-cell
    #     if 'Population' in self.datasets:
    #
    #         # The three inter datasets
    #         for index in range(7):
    #             self.calculate_MI_and_save_heatmap(self.category_and_joint_prob_per_dataset_dictionary['inter_{}'.format(index)]['categories_df'][0],
    #                 self.category_and_joint_prob_per_dataset_dictionary['inter_{}'.format(index)]['categories_df'][1], bins_on_side, base_of_log, precision, 'Inter-Generations {}'.format(index),
    #                 self.what_to_subtract, self._variable_names, self.mom_variable_symbol, self.daughter_variable_symbol, mult_number, self.what_to_subtract)
    #
    #         # do for same-cell
    #         self.calculate_MI_and_save_heatmap(self.category_and_joint_prob_per_dataset_dictionary['same_cell']['categories_df'][0],
    #             self.category_and_joint_prob_per_dataset_dictionary['same_cell']['categories_df'][1], bins_on_side, base_of_log, precision, 'Same-Cell', self.what_to_subtract, self._variable_names,
    #             self.same_cell_variable_symbol, self.same_cell_variable_symbol, mult_number, self.what_to_subtract, half_matrix=True)
    #
    #
    # def script_to_output_bootstrap(self, **kwargs):
    #
    #     how_many_times_to_resample = kwargs.get('how_many_times_to_resample', 200)
    #     # mult_number = kwargs.get('mult_number', 100000)  # to get the numbers neatly on the heatmaps
    #     precision = kwargs.get('precision', 2)  # for the joint distributions to add up to 1 and this sets how many sig-digs are shown in the heatmap
    #     bins_on_side = kwargs.get('bins_on_side', 1)  # for this bins_on_side method
    #     base_of_log = 2 * bins_on_side
    #     what_to_bootstrap = kwargs.get('what_to_bootstrap', None)
    #     percent_to_cover = kwargs.get('percent_to_cover', .9)
    #     calculate_acceleration = kwargs.get('calculate_acceleration', False)
    #     show_histograms = kwargs.get('show_histograms', False)
    #     knns = kwargs.get('knns', False)
    #
    #     # create the folder if not created already
    #     try:
    #         # Create target Directory
    #         os.mkdir(self.what_to_subtract)
    #         print("Directory ", self.what_to_subtract, " Created ")
    #     except FileExistsError:
    #         print("Directory ", self.what_to_subtract, " already exists")
    #
    #     if knns:
    #         print('knns is True')
    #         A_df = kwargs.get('A_df')
    #         B_df = kwargs.get('B_df')
    #         dset = kwargs.get('dset', None)
    #         bootstrap_dictionaries = self.bootstrap_bias_corrected_accelerated_confidence_intervals(A_df=A_df,
    #             B_df=B_df, variable_names=self._variable_names, how_many_times_to_resample=how_many_times_to_resample,
    #             percent_to_cover=percent_to_cover, calculate_acceleration=calculate_acceleration, bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision,
    #             show_histograms=show_histograms, knns=knns)
    #
    #         bar_plot_necessaries = {key: {'bounds': bootstrap_dictionaries['confidence_intervals'][key], 'MI': bootstrap_dictionaries['original'][key]} for key in
    #                                 bootstrap_dictionaries['original'].keys()}
    #
    #         sns.set_style('whitegrid')
    #         plt.bar(x=np.arange(len(bar_plot_necessaries)), height=[val['MI'][0] for val in bar_plot_necessaries.values()], yerr=np.array([val['bounds'] for val in bar_plot_necessaries.values()]).T,
    #             tick_label=list(bar_plot_necessaries.keys()), capsize=2, capstyle='butt')
    #         plt.tick_params(axis='x', rotation=90)
    #         plt.title('{} Mutual Information values with {}% confidence intervals from bootstrapping {} times'.format(dset, int(percent_to_cover * 100), how_many_times_to_resample))
    #         plt.tight_layout()
    #         plt.show()
    #         # plt.savefig('{} Mutual Information values with {}% confidence intervals from bootstrapping {} times'.format(dset, int(percent_to_cover * 100), how_many_times_to_resample), dpi=300)
    #         plt.close()
    #     else:
    #         print('knns is False')
    #         # get the category_and_joint_prob_per_dataset_dictionary
    #         self.get_the_category_and_joint_prob_for_datasets()
    #
    #         # print all the bootstrapped things you want
    #         for dset in what_to_bootstrap: # dset must be corresponding keys of the category_and_joint_prob_per_dataset_dictionary
    #             bootstrap_dictionaries = self.bootstrap_bias_corrected_accelerated_confidence_intervals(A_df=self.category_and_joint_prob_per_dataset_dictionary[dset]['categories_df'][0],
    #                 B_df=self.category_and_joint_prob_per_dataset_dictionary[dset]['categories_df'][1], variable_names=self._variable_names, how_many_times_to_resample=how_many_times_to_resample,
    #                 percent_to_cover=percent_to_cover, calculate_acceleration=calculate_acceleration, bins_on_side=bins_on_side, base_of_log=base_of_log, precision=precision,
    #                 show_histograms=show_histograms)
    #
    #             bar_plot_necessaries = {key: {'bounds': bootstrap_dictionaries['confidence_interval'][key], 'MI': bootstrap_dictionaries['original'][key]} for key in bootstrap_dictionaries['original'].keys()}
    #
    #             sns.set_style('whitegrid')
    #             plt.bar(x=np.arange(len(bar_plot_necessaries)), height=[val['MI'][0] for val in bar_plot_necessaries.values()], yerr=np.array([val['bounds'] for val in bar_plot_necessaries.values()]).T,
    #                 tick_label=list(bar_plot_necessaries.keys()), capsize=2, capstyle='butt')
    #             plt.tick_params(axis='x', rotation=90)
    #             plt.title('{} Mutual Information values with {}% confidence intervals from bootstrapping {} times'.format(dset, int(percent_to_cover * 100), how_many_times_to_resample))
    #             plt.tight_layout()
    #             plt.show()
    #             # plt.savefig('{} Mutual Information values with {}% confidence intervals from bootstrapping {} times'.format(dset, int(percent_to_cover * 100), how_many_times_to_resample), dpi=300)
    #             plt.close()

    """ This is to be used once the data is processed """

    def after_grouped_barplots_with_joint_prob_heatmap_measurements(self, **kwargs):

        def show_the_joint_dists(joint_probs, A_vars, B_vars, bins_on_side, dataset, directory, precision):
            # the filename of the plot
            filename = dataset + ' joint probabilities'



            # key corresponds to what pairing and value is the joint probability, here we have the x/y labels in key and the joint probability in val
            heatmaps = [[key, val] for key, val in joint_probs.items() if (key.split('__')[1] in A_vars) and (key.split('__')[3] in B_vars)]

            print(heatmaps)

            # A variables on the right
            fig, axes = plt.subplots(ncols=len(A_vars), nrows=len(B_vars), sharey='row', sharex='col', figsize=(12.7, 7.5))
            index = 0
            for ax in axes.flatten():
                # print(heatmaps[index])
                key = heatmaps[index][0].split('__')
                df = pd.DataFrame(columns=np.arange(2 * bins_on_side), index=np.arange(2 * bins_on_side), dtype=float)
                for col in df.columns:
                    for ind in df.index:
                        df[col].loc[ind] = heatmaps[index][1]['{}_{}'.format(col, ind)]
                sns.heatmap(data=df, ax=ax, annot=True, vmin=0, vmax=1, cbar=False, fmt='.{}f'.format(precision))  # xticklabels=np.arange(bins_on_side), yticklabels=np.arange(bins_on_side)
                # ax.set_title(self.A_variable_symbols[key[1]]+' '+self.B_variable_symbols[key[3]])
                index += 1

            for ax, row in zip(axes[:, 0], [self.A_variable_symbols[heatmaps[ind][0].split('__')[1]] for ind in np.arange(0, index, len(B_vars))]):
                ax.set_ylabel(row, rotation=0, size='large')

            for ax, col in zip(axes[0], [self.B_variable_symbols[heatmaps[ind][0].split('__')[3]] for ind in range(index)]):
                ax.set_title(col)

            plt.suptitle(filename)
            plt.tight_layout(pad=.3, rect=(0, 0, 1, .96))  # rect=(0, 0, 1, .97) rect=(0, 0.03, 1, .97),
            plt.show()
            # plt.savefig(directory + '/' + filename, dpi=300)
            plt.close()

        # joint_probs = ['Sisters', 'Nonsisters', 'Control', 'Same_cell', 'Inter_0', ..., 'Intra_0_sis/non_sis/con', ...]
        # mutual_informations = ['3 datasets', 'Inter_0', 'Inter_1', ..., 'Same_cell', '3 datasets Intra_0', '3 datasets Intra_1', ...]

        mult_number = kwargs.get('mult_number', 100000)  # to get the numbers neatly on the heatmaps
        precision = kwargs.get('precision', 2)  # for the joint distributions to add up to 1 and this sets how many sig-digs are shown in the heatmap
        bins_on_side = kwargs.get('bins_on_side', 1)  # for this bins_on_side method
        base_of_log = 2 * bins_on_side

        # get the category_and_joint_prob_per_dataset_dictionary
        self.get_the_category_and_joint_prob_for_datasets()

        # create the folder if not created already
        try:
            # Create target Directory
            os.mkdir(self.what_to_subtract)
            print("Directory ", self.what_to_subtract, " Created ")
        except FileExistsError:
            print("Directory ", self.what_to_subtract, " already exists")

        # plot the joint probability for all possible
        for key in self.category_and_joint_prob_per_dataset_dictionary.keys():
            show_the_joint_dists(self.category_and_joint_prob_per_dataset_dictionary[key]['joint_probs_dict'], ['generationtime', 'length_birth', 'growth_rate'],
                ['generationtime', 'length_birth', 'growth_rate'], bins_on_side, dataset=key, directory=self.what_to_subtract, precision=precision)
            # self.show_the_joint_dists(self.category_and_joint_prob_per_dataset_dictionary[key]['joint_probs_dict'], self._variable_names, self._variable_names, bins_on_side=bins_on_side,
            #     mean=self.what_to_subtract, dataset=key, directory=self.what_to_subtract, precision=precision)


    def __init__(self, **kwargs):

        # Generational data
        self.gen_dicts_and_class_stats(**kwargs)
        
        # Protein data
        self.get_protein_raw_data(**kwargs)


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
    attributes_table.to_csv(path_or_buf=path + 'NewSisterCellClass_attributes.csv', index=False)
    methods_table.to_csv(path_or_buf=path + 'NewSisterCellClass_methods.csv', index=False)


# def save_to_pickle_and_output_tables():
#     # For Mac
#     infiles_sisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/SISTERS/*.xls')
#     infiles_nonsisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/NONSISTERS/*.xls')
#     infiles_sisters_protein = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/ALLSIS/*.xls')
#     infiles_nonsisters_protein = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/AllNONSIS/*.xls')
#
#     print('GETTING POPULATION')
#
#     Population = ssc.Population(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
#         infiles_nonsister_protein=infiles_nonsisters_protein)
#
#     print('GETTING SISTER')
#
#     Sister = ssc.Sister(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
#         infiles_nonsister_protein=infiles_nonsisters_protein, bare_minimum_pop=True)
#
#     print('GETTING NONSISTER')
#
#     Nonsister = ssc.Nonsister(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
#         infiles_nonsister_protein=infiles_nonsisters_protein, bare_minimum_pop=True)
#
#     print('GETTING CONTROL')
#
#     Control = ssc.Control(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', bare_minimum_pop=True, bare_minimum_sis=True,
#         bare_minimum_non_sis=True, infiles_sister_protein=infiles_sisters_protein, infiles_nonsister_protein=infiles_nonsisters_protein)
#
#     print('GETTING ENVIRONMENTAL SISTERS')
#
#     Env_Sister = ssc.Env_Sister(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
#         infiles_nonsister_protein=infiles_nonsisters_protein, bare_minimum_pop=True)
#
#     print('GETTING ENVIRONMENTAL NONSISTERS')
#
#     Env_Nonsister = ssc.Env_Nonsister(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
#         infiles_nonsister_protein=infiles_nonsisters_protein, bare_minimum_pop=True)
#
#     print('OUTPUTTING THE ATTRIBUTES AND METHODS FOR THE CLASSES TO CSV')
#     output_the_attributes_and_methods_of_classes(Population, Sister, Nonsister, Control, r'/Users/alestawsky/PycharmProjects/untitled/')
#
#     # Create many pickles and dump the respective classes into them
#     pickle_out = open("NewSisterCellClass_Population.pickle", "wb")
#     pickle.dump(Population, pickle_out)
#     pickle_out.close()
#
#     pickle_out = open("NewSisterCellClass_Sister.pickle", "wb")
#     pickle.dump(Sister, pickle_out)
#     pickle_out.close()
#
#     pickle_out = open("NewSisterCellClass_Nonsister.pickle", "wb")
#     pickle.dump(Nonsister, pickle_out)
#     pickle_out.close()
#
#     pickle_out = open("NewSisterCellClass_Control.pickle", "wb")
#     pickle.dump(Control, pickle_out)
#     pickle_out.close()
#
#     pickle_out = open("NewSisterCellClass_Env_Sister.pickle", "wb")
#     pickle.dump(Env_Sister, pickle_out)
#     pickle_out.close()
#
#     pickle_out = open("NewSisterCellClass_Env_Nonsister.pickle", "wb")
#     pickle.dump(Env_Nonsister, pickle_out)
#     pickle_out.close()
#
#     print('SAVED TO PICKLE')


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

    """ Plotting the histograms of the coefficient of covariance """  # Sister.plot_hist_of_coefs_of_variance()  #  # Nonsister.plot_hist_of_coefs_of_variance()  #  # Control.plot_hist_of_coefs_of_variance()


def save_the_median_data(infiles_sisters, infiles_nonsisters, infiles_sisters_protein, infiles_nonsisters_protein):
    # ONLY FOR GLOBAL MEAN AND MEDIAN IS GENERATIONTIME DISCRETE, ONCE THE TRAP IS SUBTRACTED THE CONTINUOSITY OF THE FIRST MOMENT DISRUPTS THE DISCRETENESS, SHOULD WE CHANGE THIS?

    measurements = SisterCellData(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
        infiles_nonsister_protein=infiles_nonsisters_protein, what_to_subtract=None, start_index=None, end_index=None, datasets=['Population', 'Sisters', 'Nonsisters', 'Control'])

    pickle_out = open("measurements.pickle", "wb")
    pickle.dump(measurements, pickle_out)
    pickle_out.close()

    print('!!!!!!!!measurements saved!!!!!!!!!!!')

    pop_median = SisterCellData(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
        infiles_nonsister_protein=infiles_nonsisters_protein, what_to_subtract='global median', start_index=None, end_index=None, datasets=['Population', 'Sisters', 'Nonsisters', 'Control'])

    pickle_out = open("pop_median.pickle", "wb")
    pickle.dump(pop_median, pickle_out)
    pickle_out.close()

    print('!!!!!!!!pop_median saved!!!!!!!!!!!')

    trap_median = SisterCellData(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
        infiles_nonsister_protein=infiles_nonsisters_protein, what_to_subtract='trap median', start_index=None, end_index=None, datasets=['Population', 'Sisters', 'Nonsisters', 'Control'])

    pickle_out = open("trap_median.pickle", "wb")
    pickle.dump(trap_median, pickle_out)
    pickle_out.close()

    print('!!!!!!!!!!!trap_median saved!!!!!!!!!')

    traj_median = SisterCellData(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
        infiles_nonsister_protein=infiles_nonsisters_protein, what_to_subtract='traj median', start_index=None, end_index=None, datasets=['Population', 'Sisters', 'Nonsisters', 'Control'])

    pickle_out = open("traj_median.pickle", "wb")
    pickle.dump(traj_median, pickle_out)
    pickle_out.close()

    print('!!!!!!!traj_median saved!!!!!!!!!!')


def reason_why_gentime_cant_be_half_on_both_sides():
    # import them
    pickle_in = open("measurements.pickle", "rb")
    measurements = pickle.load(pickle_in)
    pickle_in.close()

    something = np.cumsum([len(measurements.all_bacteria[measurements.all_bacteria['generationtime'] == val]) / len(measurements.all_bacteria['generationtime']) for val in
                           np.sort(np.unique(measurements.all_bacteria['generationtime']))])
    print(something)
    gentime_dict = dict(zip(np.sort(np.unique(measurements.all_bacteria['generationtime'])), something))
    print(gentime_dict)
    plt.step(gentime_dict.keys(), gentime_dict.values())
    plt.xlabel('generationtime')
    plt.ylabel('CDF')
    plt.title('All Cells Recorded')
    plt.axhline(.5, color='black', ls='--')
    plt.show()


def bootstrap_for_continuous(data, A_df, B_df, variables):
    def discrete_binning_method_kmeans(A_df, B_df, log_A_df, log_B_df, bins_in_total, variable_names, log_vars):
        df_A_cut = pd.DataFrame(columns=variable_names)
        df_B_cut = pd.DataFrame(columns=variable_names)
        for var in variable_names:
            if var in log_vars:
                A_centered = np.array(log_A_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]).reshape(-1, 1)
                B_centered = np.array(log_B_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]).reshape(-1, 1)
            else:
                A_centered = np.array(A_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]).reshape(-1, 1)
                B_centered = np.array(B_df[var].iloc[:min(len(A_df[var]), len(B_df[var]))]).reshape(-1, 1)
            joint_centered = np.concatenate((A_centered, B_centered), axis=0)

            est = KBinsDiscretizer(n_bins=bins_in_total, encode='ordinal', strategy='kmeans')
            est.fit(joint_centered)

            edges = np.sort(est.bin_edges_[0])

            # based on the bin edge, give each entry in the A and B series an integer that maps to the bin in the marginal distribution
            A_cut = pd.cut(A_centered.flatten(), edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))  # , duplicates='drop'
            B_cut = pd.cut(B_centered.flatten(), edges, right=True, include_lowest=True, labels=np.arange(len(edges) - 1))

            print('number of bins:', est.n_bins)
            if est.n_bins != bins_in_total:
                raise IOError('est.n_bins != bins_in_total')

            # append this trap's variable's labels
            df_A_cut[var] = A_cut
            df_B_cut[var] = B_cut

        return df_A_cut, df_B_cut

    # data.discrete_binning_method_kmeans(A_df=data.sis_A_pooled, B_df, log_A_df, log_B_df, bins_in_total, variable_names, log_vars)
    # for var in variables:




if __name__ == '__main__':
    # For Mac
    infiles_sisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/SISTERS/*.xls')
    infiles_nonsisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/NONSISTERS/*.xls')
    infiles_sisters_protein = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/ALLSIS/*.xls')
    infiles_nonsisters_protein = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/AllNONSIS/*.xls')

    # measurements = SisterCellData(infiles_sister=infiles_sisters, infiles_nonsister=infiles_nonsisters, discretization_variable='length', infiles_sister_protein=infiles_sisters_protein,
    #     infiles_nonsister_protein=infiles_nonsisters_protein, what_to_subtract=None, start_index=None, end_index=None, datasets=['Population', 'Sisters', 'Nonsisters', 'Control'])
    #
    # pickle_out = open("measurements.pickle", "wb")
    # pickle.dump(measurements, pickle_out)
    # pickle_out.close()
    #
    # print('!!!!!!!!measurements saved!!!!!!!!!!!')
    # exit()

    # import them
    pickle_in = open("measurements.pickle", "rb")
    measurements = pickle.load(pickle_in)
    pickle_in.close()

    # gentimes = {}
    # smaller_alphas = {}
    # for ind in range(60): # for smaller alpha index
    #     print(ind)
    #     print('----')
    #     for val in measurements.pop_inside_generation_A_dict.values(): # for every bacteria we have in the A traces of S and NS which is half of all bacteria
    #         if len(val['growth_rates']) > ind:
    #             # print(val['growth_rates'])
    #             for array in val['growth_rates']:
    #                 if len(array) <= ind:
    #                     print(array)
    #                     print('smaller!')
    #                 else:
    #                     # print(array[-1])
    #                     pass
    #             print(val['generationtime'].shape)
    #             # print(val['growth_rates'])
    #             print(len([array[ind] for array in val['growth_rates']]))
    #
    #         # if len(val) > ind:
    #         #     print(val['growth_rates'][ind])
    #         #     print(val['generationtime'][ind])
    #         else:
    #             continue
    # exit()
    
    def group_them_into_same_size_arrays(dict1, dict2, inside_number):
        smaller_alphas_A = dict(zip(np.arange(inside_number), [[] for n in np.arange(inside_number)]))
        gentimes_A = dict(zip(np.arange(inside_number), [[] for n in np.arange(inside_number)]))
        smaller_alphas_B = dict(zip(np.arange(inside_number), [[] for n in np.arange(inside_number)]))
        gentimes_B = dict(zip(np.arange(inside_number), [[] for n in np.arange(inside_number)]))

        # inside_var

        if dict1.keys() != dict2.keys():
            raise IOError("they don't have the same keys!")

        # for key in dict1.keys():

        #     if len(dict1[key]['generationtime']) != len(dict2[key]['generationtime']):
        #         raise IOError("They don't have the same amount of generations")
        #     else:
        #         for gen_index in range(len(dict1[key]['generationtime'])):
        #             if len(dict1[key][inside_var][gen_index]) > inside_number:
        #
        #             else:
        #                 continue
        #             if len(dict1[key]['growth_rate'][gen_index]) != len(dict2[key]['growth_rate'][gen_index]):
        #                 print("they don't have the same amount of inside steps, ie. non-equal generationtimes")
        #                 min_insides = min(len(dict1[key]['growth_rate'][gen_index]), len(dict2[key]['growth_rate'][gen_index]))
        #
        # if len(val1['generationtime']) != len(val2['generationtime']):
        #     print('not the same amount of generations in ')
        #     min_gen = min(len(val1['generationtime']), len(val2['generationtime']))
        # for ind in np.arange(inside_number):  # for the windows inside the generation
        #
        #     for val1, val2 in zip(dict1.values(), dict2.values()):
        #         # if len(val1['growth_rates']) > ind: # if the trap is big enough to contain the n-th window we are asking for
        #
        #         for gen_ind in
        #
        #         if len(val1['generationtime']) != len(val2['growth_rates']):  # checking to see there are the same amount of generations in both
        #             print(len(val1['generationtime']), len(val2['growth_rates']), 'not equal :(')
        #             exit()
        #
        #         for gen_index in range(len(val1['generationtime'])):  # for all the generations inside this trap
        #             if len(val1['growth_rates'][gen_index]) > ind:  # if the generation is big enough to contain the n-th window we are asking for
        #                 smaller_alphas_A[ind].append(val1['growth_rates'][gen_index][ind])
        #                 gentimes_A[ind].append(val1['generationtime'][gen_index])
        #
        #     for val in dict2.values():  # for all the traps
        #
        #         # if len(val['growth_rates']) > ind: # if the trap is big enough to contain the n-th window we are asking for
        #
        #         if len(val['generationtime']) != len(val['growth_rates']):  # checking to see there are the same amount of generations in both
        #             print(len(val['generationtime']), len(val['growth_rates']), 'not equal :(')
        #             exit()
        #
        #         for gen_index in range(len(val['generationtime'])):  # for all the generations inside this trap
        #             if len(val['growth_rates'][gen_index]) > ind:  # if the generation is big enough to contain the n-th window we are asking for
        #                 smaller_alphas_B[ind].append(val['growth_rates'][gen_index][ind])
        #                 gentimes_B[ind].append(val['generationtime'][gen_index])

        return smaller_alphas_A, smaller_alphas_B, gentimes_A, gentimes_B
    
    # def group_them_into_same_size_arrays(dict1, dict2, inside_number):
    #     smaller_alphas_A = dict(zip(np.arange(inside_number), [[] for n in np.arange(inside_number)]))
    #     gentimes_A = dict(zip(np.arange(inside_number), [[] for n in np.arange(inside_number)]))
    #     smaller_alphas_B = dict(zip(np.arange(inside_number), [[] for n in np.arange(inside_number)]))
    #     gentimes_B = dict(zip(np.arange(inside_number), [[] for n in np.arange(inside_number)]))
    #     for ind in np.arange(inside_number):  # for the windows inside the generation
    #         for val in dict1.values():  # for all the traps
    # 
    #             # if len(val['growth_rates']) > ind: # if the trap is big enough to contain the n-th window we are asking for
    # 
    #             if len(val['generationtime']) != len(val['growth_rates']):  # checking to see there are the same amount of generations in both
    #                 print(len(val['generationtime']), len(val['growth_rates']), 'not equal :(')
    #                 exit()
    # 
    #             for gen_index in range(len(val['generationtime'])):  # for all the generations inside this trap
    #                 if len(val['growth_rates'][gen_index]) > ind:  # if the generation is big enough to contain the n-th window we are asking for
    #                     smaller_alphas_A[ind].append(val['growth_rates'][gen_index][ind])
    #                     gentimes_A[ind].append(val['generationtime'][gen_index])
    # 
    #         for val in dict2.values():  # for all the traps
    # 
    #             # if len(val['growth_rates']) > ind: # if the trap is big enough to contain the n-th window we are asking for
    # 
    #             if len(val['generationtime']) != len(val['growth_rates']):  # checking to see there are the same amount of generations in both
    #                 print(len(val['generationtime']), len(val['growth_rates']), 'not equal :(')
    #                 exit()
    # 
    #             for gen_index in range(len(val['generationtime'])):  # for all the generations inside this trap
    #                 if len(val['growth_rates'][gen_index]) > ind:  # if the generation is big enough to contain the n-th window we are asking for
    #                     smaller_alphas_B[ind].append(val['growth_rates'][gen_index][ind])
    #                     gentimes_B[ind].append(val['generationtime'][gen_index])
    # 
    #     return smaller_alphas_A, smaller_alphas_B, gentimes_A, gentimes_B


    smaller_alphas_A, smaller_alphas_B, gentimes_A, gentimes_B = group_them_into_same_size_arrays(measurements.sis_inside_generation_A_dict, measurements.sis_inside_generation_B_dict, 20)
    
    smaller_alphas_1, smaller_alphas_2, gentimes_1, gentimes_2 = group_them_into_same_size_arrays(measurements.non_sis_inside_generation_A_dict, measurements.non_sis_inside_generation_B_dict, 20)

    pearson_array_A = []
    x_array_A = []
    for key in smaller_alphas_A.keys():
        print(key)
        print(len(smaller_alphas_A[key]), len(smaller_alphas_B[key]))
        if (len(smaller_alphas_A[key]) > 1) and (len(gentimes_A[key]) > 1):
            pearson_array_A.append(stats.pearsonr(smaller_alphas_A[key], smaller_alphas_B[key])[0])
            x_array_A.append(key)

    pearson_array_B = []
    x_array_B = []
    for key in smaller_alphas_B.keys():
        print(key)
        print(smaller_alphas_B[key], gentimes_B[key])
        if (len(smaller_alphas_B[key]) > 1) and (len(gentimes_B[key]) > 1):
            pearson_array_B.append(stats.pearsonr(gentimes_A[key], gentimes_B[key])[0])
            x_array_B.append(key)

    pearson_array_1 = []
    x_array_1 = []
    for key in smaller_alphas_1.keys():
        print(key)
        print(smaller_alphas_1[key], gentimes_1[key])
        if (len(smaller_alphas_1[key]) > 1) and (len(gentimes_1[key]) > 1):
            pearson_array_1.append(stats.pearsonr(smaller_alphas_1[key], smaller_alphas_2[key])[0])
            x_array_1.append(key)

    pearson_array_2 = []
    x_array_2 = []
    for key in smaller_alphas_2.keys():
        print(key)
        print(smaller_alphas_2[key], gentimes_2[key])
        if (len(smaller_alphas_2[key]) > 1) and (len(gentimes_2[key]) > 1):
            pearson_array_2.append(stats.pearsonr(gentimes_1[key], gentimes_2[key])[0])
            x_array_2.append(key)

    plt.plot(x_array_A, pearson_array_A, marker='.', label='sisA smaller alphas')
    plt.plot(x_array_B, pearson_array_B, marker='.', label='sisB')
    plt.plot(x_array_1, pearson_array_1, marker='.', label='nonsisA smaller alphas')
    plt.plot(x_array_2, pearson_array_2, marker='.', label='nonsisB')
    plt.show()
    plt.close()

    exit()


    smaller_alphas = np.array(
        [np.array([np.array([array[ind] for array in val['growth_rates'] if len(array) > ind]) for val in measurements.pop_inside_generation_A_dict.values() if len(val['growth_rates']) > ind]) for ind
            in range(60)])
    gentimes = np.array([np.array([np.array([val['generationtime'][gen_index] for gen_index in range(len(val['generationtime'])) if len(val['growth_rates'][gen_index]) > ind]) for val in
                                   measurements.pop_inside_generation_A_dict.values() if len(val['growth_rates']) > ind]) for ind in range(60)])
    # print(smaller_alphas.shape)

    pearson_array = []
    for smaller_alpha, gentime in zip(smaller_alphas, gentimes):
        print(len(smaller_alpha))
        print(len(gentime))
        sa_array = np.array([])
        gt_array = np.array([])
        for sa, gt in zip(smaller_alpha, gentime):

            if len(sa) != len(gt):
                print(sa, gt)
                print(len(sa), len(gt))
                exit()
            sa_array = np.append(sa_array, sa)
            gt_array = np.append(gt_array, gt)
        # for sa, gt in zip(smaller_alpha, gentime):
        #     print(measurements.pop_inside_generation_A_dict['0']['generationtime'])
        #     print(sa.shape)
        #     print(gt.shape)
        #     print(sa)
        #     print(gt)
        #     exit()

        print(sa_array)
        print(gt_array)
        print(sa_array.shape)
        print(gt_array.shape)
        exit()

        pearson_array.append(stats.pearsonr(sa_array, gt_array)[0])

        # plt.plot([stats.pearsonr(smaller_alpha, gentime)[0] for smaller_alpha, gentime in zip(smaller_alphas, gentimes)])
        # plt.show()
        # plt.close()

    plt.plot(np.arange(len(pearson_array)), pearson_array)
    plt.show()
    plt.close()

    # smaller_alphas = np.array([np.array([np.array([array[ind] if len(array) > ind else array[-1] for array in val['growth_rates']]) if len(val['growth_rates']) > ind]) for val in measurements.pop_inside_generation_A_dict.values() for ind in range(60)])
    gentimes = np.array([np.array([np.array(val['generationtime']) for val in measurements.pop_inside_generation_A_dict.values() if len(val['growth_rates']) > ind]) for ind in range(60)])

    print(smaller_alphas)
    print(gentimes)

    # smaller_alpha_array = np.array([np.array([val['growth_rates'][ind] for val in measurements.pop_inside_generation_A_dict.values() if len(val) > ind]) for ind in range(60)])
    # print(smaller_alpha_array.shape)
    # gentime_array = np.array([np.array([val['generationtime'][ind] for val in measurements.pop_inside_generation_A_dict.values() if len(val) > ind]) for ind in range(60)])
    # print(gentime_array.shape)
    #

    #
    # sns.distplot([len(val['growth_rates']) for val in measurements.sis_inside_generation_A_dict.values()])
    # plt.show()
    # plt.close()

    print(len(measurements.sis_inside_generation_A_dict.keys()))
    print(np.max([len(val['growth_rates']) for val in measurements.sis_inside_generation_A_dict.values()]))
    exit()

    # mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
    print('sis')
    print('score', mutual_info_score(measurements.sis_A_pooled['generationtime'], measurements.sis_B_pooled['generationtime']), 'adjusted',
        adjusted_mutual_info_score(measurements.sis_A_pooled['generationtime'], measurements.sis_B_pooled['generationtime']), 'normalized',
        normalized_mutual_info_score(measurements.sis_A_pooled['generationtime'], measurements.sis_B_pooled['generationtime']))
    print('non_sis')
    print('score', mutual_info_score(measurements.non_sis_A_pooled['generationtime'], measurements.non_sis_B_pooled['generationtime']), 'adjusted',
        adjusted_mutual_info_score(measurements.non_sis_A_pooled['generationtime'], measurements.non_sis_B_pooled['generationtime']), 'normalized',
        normalized_mutual_info_score(measurements.non_sis_A_pooled['generationtime'], measurements.non_sis_B_pooled['generationtime']))
    print('con')
    print('score', mutual_info_score(measurements.con_A_pooled['generationtime'], measurements.con_B_pooled['generationtime']), 'adjusted',
        adjusted_mutual_info_score(measurements.con_A_pooled['generationtime'], measurements.con_B_pooled['generationtime']), 'normalized',
        normalized_mutual_info_score(measurements.con_A_pooled['generationtime'], measurements.con_B_pooled['generationtime']))

    print('sis intragen 0')
    print('score', mutual_info_score(measurements.sis_A_intra_gen_bacteria[0]['generationtime'], measurements.sis_B_intra_gen_bacteria[0]['generationtime']), 'adjusted',
        adjusted_mutual_info_score(measurements.sis_A_intra_gen_bacteria[0]['generationtime'], measurements.sis_B_intra_gen_bacteria[0]['generationtime']), 'normalized',
        normalized_mutual_info_score(measurements.sis_A_intra_gen_bacteria[0]['generationtime'], measurements.sis_B_intra_gen_bacteria[0]['generationtime']))
    print('non_sis intragen 0')
    print('score', mutual_info_score(measurements.non_sis_A_intra_gen_bacteria[0]['generationtime'], measurements.non_sis_B_intra_gen_bacteria[0]['generationtime']), 'adjusted',
        adjusted_mutual_info_score(measurements.non_sis_A_intra_gen_bacteria[0]['generationtime'], measurements.non_sis_B_intra_gen_bacteria[0]['generationtime']), 'normalized',
        normalized_mutual_info_score(measurements.non_sis_A_intra_gen_bacteria[0]['generationtime'], measurements.non_sis_B_intra_gen_bacteria[0]['generationtime']))
    print('con intragen 0')
    print('score', mutual_info_score(measurements.con_A_intra_gen_bacteria[0]['generationtime'], measurements.con_B_intra_gen_bacteria[0]['generationtime']), 'adjusted',
        adjusted_mutual_info_score(measurements.con_A_intra_gen_bacteria[0]['generationtime'], measurements.con_B_intra_gen_bacteria[0]['generationtime']), 'normalized',
        normalized_mutual_info_score(measurements.con_A_intra_gen_bacteria[0]['generationtime'], measurements.con_B_intra_gen_bacteria[0]['generationtime']))

    x_range = np.arange(len(measurements.sis_A_intra_gen_bacteria))
    # plt.plot(x_range, [mutual_info_score(measurements.sis_A_intra_gen_bacteria[index]['generationtime'], measurements.sis_B_intra_gen_bacteria[index]['generationtime']) for index in x_range], label='score', marker='.', ls=':', color='blue')
    # plt.plot(x_range, [adjusted_mutual_info_score(measurements.sis_A_intra_gen_bacteria[index]['generationtime'], measurements.sis_B_intra_gen_bacteria[index]['generationtime']) for index in x_range], label='adjusted',
    #     marker='.', ls='--', color='blue')
    plt.plot(x_range, [adjusted_mutual_info_score(measurements.sis_A_intra_gen_bacteria[index]['generationtime'], measurements.sis_B_intra_gen_bacteria[index]['generationtime']) for index in x_range], label='adjusted',
        marker='.', ls='-', color='blue')
    plt.plot(x_range,
        [adjusted_mutual_info_score(measurements.non_sis_A_intra_gen_bacteria[index]['generationtime'], measurements.non_sis_B_intra_gen_bacteria[index]['generationtime']) for index in x_range],
        label='adjusted', marker='.', ls='-', color='orange')
    plt.plot(x_range,
        [adjusted_mutual_info_score(measurements.con_A_intra_gen_bacteria[index]['generationtime'], measurements.con_B_intra_gen_bacteria[index]['generationtime']) for index in x_range],
        label='adjusted', marker='.', ls='-', color='green')
    plt.xlabel('intra generation')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.show()
    plt.close()

    x_range = np.arange(len(measurements.mother_dfs))
    # plt.plot(x_range, [mutual_info_score(measurements.mother_dfs[index]['generationtime'], measurements.daughter_dfs[index]['generationtime']) for index in x_range], label='score', marker='.')
    plt.plot(x_range, [adjusted_mutual_info_score(measurements.mother_dfs[index]['generationtime'], measurements.daughter_dfs[index]['generationtime']) for index in x_range], label='adjusted', marker='.')
    # plt.plot(x_range, [normalized_mutual_info_score(measurements.mother_dfs[index]['generationtime'], measurements.daughter_dfs[index]['generationtime']) for index in x_range], label='normalized', marker='.')
    plt.xlabel('intra generation')
    plt.ylabel('Mutual Information')
    plt.legend()
    plt.show()
    plt.close()

    print('inter gen 0')
    print('score', mutual_info_score(measurements.mother_dfs[0]['generationtime'], measurements.daughter_dfs[0]['generationtime']), 'adjusted',
        adjusted_mutual_info_score(measurements.mother_dfs[0]['generationtime'], measurements.daughter_dfs[0]['generationtime']), 'normalized',
        normalized_mutual_info_score(measurements.mother_dfs[0]['generationtime'], measurements.daughter_dfs[0]['generationtime']))
    
    exit()

    # pop_median.after_grouped_barplots_with_joint_prob_heatmap()

    # get the category_and_joint_prob_per_dataset_dictionary
    measurements.get_the_category_and_joint_prob_for_datasets()

    print(measurements.category_and_joint_prob_per_dataset_dictionary.keys())


    # pickle_in = open("trap_median.pickle", "rb")
    # trap_median = pickle.load(pickle_in)
    # pickle_in.close()
    # pickle_in = open("traj_median.pickle", "rb")
    # traj_median = pickle.load(pickle_in)
    # pickle_in.close()











    #
    # # print('LOOK AT THIS', len(np.unique(data.all_bacteria['generationtime'])), np.unique(data.all_bacteria['generationtime']))
    # # plt.hist(data.all_bacteria['generationtime'], bins=len(np.unique(data.all_bacteria['generationtime'])), density=True)
    # # plt.show()
    # # plt.close()
    #
    # # data.script_to_output_bootstrap_bins_on_side(A_df=data.sis_A_pooled, B_df=data.sis_B_pooled, how_many_times_to_resample=500, percent_to_cover=.9, calculate_acceleration=False, show_histograms=False, knns=50)
    # # data.script_to_output_bootstrap_bins_on_side(what_to_bootstrap=['sis'], how_many_times_to_resample=500, percent_to_cover=.9, calculate_acceleration=False, show_histograms=False, knns=50)
    #
    # # bootstrap_dictionaries = data.bootstrap_bias_corrected_accelerated_confidence_intervals(A_df=data.sis_A_pooled, B_df=data.sis_B_pooled, variable_names=data._variable_names,
    # #     how_many_times_to_resample=500, percent_to_cover=.9, calculate_acceleration=False, bins_on_side=1, base_of_log=np.e,
    # #     precision=2, show_histograms=False, knns=50)
    # #
    # # print(len(bootstrap_dictionaries['bootstrap'][list(bootstrap_dictionaries['bootstrap'].keys())[0]]), bootstrap_dictionaries['bootstrap'][list(bootstrap_dictionaries['bootstrap'].keys())[0]])
    # # pickle_out = open("Sister_pooled_estimated_MIs_natural_log_bootstrap_dicts_{}_resamples_{}_percent_covered_k_is_{}.pickle".format(500, 90, 50), "wb")
    # # pickle.dump(bootstrap_dictionaries, pickle_out)
    # # pickle_out.close()
    # #
    # # bar_plot_necessaries = {key: {'bounds': bootstrap_dictionaries['confidence_intervals'][key], 'MI': bootstrap_dictionaries['original'][key]} for key in bootstrap_dictionaries['original'].keys()}
    # #
    # # sns.set_style('whitegrid')
    # # plt.bar(x=np.arange(len(bar_plot_necessaries)), height=[val['MI'][0] for val in bar_plot_necessaries.values()], yerr=np.array([val['bounds'] for val in bar_plot_necessaries.values()]).T,
    # #     tick_label=list(bar_plot_necessaries.keys()), capsize=2, capstyle='butt')
    # # plt.tick_params(axis='x', rotation=90)
    # # title = '{} natural_log estimated Mutual Informations with {}% confidence intervals from bootstrapping {} times'.format('Sister', int(.9 * 100), 500)
    # # plt.title(title)
    # # plt.tight_layout()
    # # plt.show()
    # # # plt.savefig('{} Mutual Information values with {}% confidence intervals from bootstrapping {} times'.format(dset, int(percent_to_cover * 100), how_many_times_to_resample), dpi=300)
    # # plt.close()
    #
    # bootstrap_dictionaries = data.bootstrap_bias_corrected_accelerated_confidence_intervals(what_to_bootstrap=['sis'], variable_names=data._variable_names,
    #     how_many_times_to_resample=500, percent_to_cover=.9, calculate_acceleration=False, bins_on_side=1, base_of_log=np.e, precision=2, show_histograms=False)
    # print(len(bootstrap_dictionaries['bootstrap'][list(bootstrap_dictionaries['bootstrap'].keys())[0]]), bootstrap_dictionaries['bootstrap'][list(bootstrap_dictionaries['bootstrap'].keys())[0]])
    # pickle_out = open("Sister_pooled_binary_MIs_natural_log_bootstrap_dicts_{}_resamples_{}_percent_covered.pickle".format(500, 90), "wb")
    # pickle.dump(bootstrap_dictionaries, pickle_out)
    # pickle_out.close()
    #
    # bar_plot_necessaries = {key: {'bounds': bootstrap_dictionaries['confidence_interval'][key], 'MI': bootstrap_dictionaries['original'][key]} for key in bootstrap_dictionaries['original'].keys()}
    #
    # sns.set_style('whitegrid')
    # plt.bar(x=np.arange(len(bar_plot_necessaries)), height=[val['MI'][0] for val in bar_plot_necessaries.values()], yerr=np.array([val['bounds'] for val in bar_plot_necessaries.values()]).T,
    #     tick_label=list(bar_plot_necessaries.keys()), capsize=2, capstyle='butt')
    # plt.tick_params(axis='x', rotation=90)
    # plt.title('{} natural_log binary Mutual Information values with {}% confidence intervals from bootstrapping {} times'.format('Sister', int(.9 * 100), 500))
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig('{} Mutual Information values with {}% confidence intervals from bootstrapping {} times'.format(dset, int(percent_to_cover * 100), how_many_times_to_resample), dpi=300)
    # plt.close()
    #
    # exit()
    # data.discrete_binning_method_kmeans(A_df=data.sis_A_pooled, B_df=data.sis_B_pooled, log_A_df=data.sis_log_A_pooled, log_B_df=data.sis_log_B_pooled, bins_in_total=15,
    #     variable_names=['length_birth', 'length_final', 'growth_rate', 'fold_growth', 'division_ratio', 'added_length'],
    #     log_vars=['generationtime', 'length_birth', 'length_final', 'growth_rate', 'fold_growth', 'added_length'])
    # exit()
    #
    # # print(A_categories, B_categories)
    # # joint_probs_dict = data.get_joint_probs_dict(A_categories, B_categories, data._variable_names, bins_on_side=1, base_of_log=2, precision=2)
    # # print(joint_probs_dict)
    # # data.show_the_joint_dists(joint_probs_dict, data._variable_names, data._variable_names, bins_on_side=1, mean='trap median', dataset='Population', directory='somewhere', precision=2)
    # # exit()
    # # creates the pickle and saves it to directory we are automatically working under
    # # save_to_pickle_and_output_tables()  # debugging_by_importing_the_data()
