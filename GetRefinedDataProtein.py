
import numpy as np

import glob

import pandas as pd

from sklearn.model_selection import train_test_split

import pickle

import SisterCellClassWithProtein as ssc

import CALCULATETHEBETAS

import CALCULATETHEBETAS1


def InitiationOnlyOnce(struct, pickle_name):
    # Get the population level statistics of cycle params to convert to Standard Normal variables
    sis_traces = [A for A in struct.dict_with_all_sister_traces.values()]
    non_traces = [B for B in struct.dict_with_all_non_sister_traces.values()]
    all_traces = sis_traces + non_traces
    # mean and std div for 'generationtime', 'length_birth', 'length_final', 'growth_length', 'division_ratios__f_n'
    pop_means = [np.mean(pd.concat(np.array([trace[c_p] for trace in all_traces]), ignore_index=True)) for c_p in all_traces[0].keys()]
    pop_stds = [np.std(pd.concat(np.array([trace[c_p] for trace in all_traces]), ignore_index=True)) for c_p in all_traces[0].keys()]
    phi_mean = np.mean(pd.concat(np.array([trace['generationtime']*trace['growth_length'] for trace in all_traces]), ignore_index=True))
    phi_std = np.std(pd.concat(np.array([trace['generationtime']*trace['growth_length'] for trace in all_traces]), ignore_index=True))
    pop_means.append(phi_mean)
    pop_stds.append(phi_std)
    print('population statistics:', pop_means, pop_stds, len(pop_means), len(pop_stds))

    # columns for the dataframe
    cols = ['generationtime_m', 'length_birth_m', 'length_final_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m',
            'generationtime_d', 'length_birth_d', 'length_final_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d',
            'trap_average_generationtime', 'trap_average_length_birth', 'trap_average_length_final',
            'trap_average_growth_length', 'trap_average_division_ratios__f_n', 'trap_average_phi']

    ''' put in the struct '''
    struct.cols = cols
    struct.pop_means = pop_means
    struct.pop_stds = pop_stds

    # Data Frame
    m_d_dependance = pd.DataFrame(columns=cols)
    m_d_dependance_units = pd.DataFrame(columns=cols)

    # ind is the index to add everything to the mother daughter dataframe
    ind = 0
    A_sis = np.array([key for key in struct.A_dict_sis.keys()])
    B_sis = np.array([key for key in struct.B_dict_sis.keys()])

    # the experiment A cell data
    for key in A_sis:
        # the dataframe with cycle params
        val = struct.A_dict_sis[key]
        val['phi'] = val['generationtime'] * val['growth_length']
        # all the cycle params' means
        trap_averages = np.mean(val)
        # turn them into a standard normal value
        trap_averages_standard = (trap_averages - pop_means) / pop_stds
        # trap_averages = (trap_averages) / pop_stds
        for row1, row2 in zip(range(len(val) - 1), range(1, len(val))):  # row1 is the previous generation of row2, ie. the mom and daughter
            mom = np.array(val.iloc[row1])
            daughter = np.array(val.iloc[row2])
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages)), index=m_d_dependance_units.columns)
            m_d_dependance_units.loc[ind] = together
            # convert them to standard normal values
            mom = (mom - pop_means) / pop_stds
            daughter = (daughter - pop_means) / pop_stds
            # mom = (mom) / pop_stds
            # daughter = (daughter) / pop_stds
            # concatenate them and add them to the dataframe
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages_standard)), index=m_d_dependance.columns)
            m_d_dependance.loc[ind] = together
            # move on to the next generation
            ind = ind + 1

    print('A sis traces finished')

    # the experiment B cell data
    for key in B_sis:
        # the dataframe with cycle params
        val = struct.B_dict_sis[key]
        val['phi'] = val['generationtime'] * val['growth_length']
        # all the cycle params' means
        trap_averages = np.mean(val)
        # turn them into a standard normal value
        trap_averages_standard = (trap_averages - pop_means) / pop_stds
        # trap_averages = (trap_averages) / pop_stds
        for row1, row2 in zip(range(len(val) - 1), range(1, len(val))):  # row1 is the previous generation of row2, ie. the mom and daughter
            mom = np.array(val.iloc[row1])
            daughter = np.array(val.iloc[row2])
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages)), index=m_d_dependance_units.columns)
            m_d_dependance_units.loc[ind] = together
            # convert them to standard normal values
            mom = (mom - pop_means) / pop_stds
            daughter = (daughter - pop_means) / pop_stds
            # mom = (mom) / pop_stds
            # daughter = (daughter) / pop_stds
            # concatenate them and add them to the dataframe
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages_standard)), index=m_d_dependance.columns)
            m_d_dependance.loc[ind] = together
            # move on to the next generation
            ind = ind + 1

    print('B sis traces finished')

    # ind is the index to add everything to the mother daughter dataframe
    A_non_sis = np.array([key for key in struct.A_dict_non_sis.keys()])
    B_non_sis = np.array([key for key in struct.B_dict_non_sis.keys()])

    # the experiment A cell data
    for key in A_non_sis:
        # the dataframe with cycle params
        val = struct.A_dict_non_sis[key]
        val['phi'] = val['generationtime'] * val['growth_length']
        # all the cycle params' means
        trap_averages = np.mean(val)
        # turn them into a standard normal value
        trap_averages_standard = (trap_averages - pop_means) / pop_stds
        # trap_averages = (trap_averages) / pop_stds
        for row1, row2 in zip(range(len(val) - 1), range(1, len(val))):  # row1 is the previous generation of row2, ie. the mom and daughter
            mom = np.array(val.iloc[row1])
            daughter = np.array(val.iloc[row2])
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages)), index=m_d_dependance_units.columns)
            m_d_dependance_units.loc[ind] = together
            # convert them to standard normal values
            mom = (mom - pop_means) / pop_stds
            daughter = (daughter - pop_means) / pop_stds
            # mom = (mom) / pop_stds
            # daughter = (daughter) / pop_stds
            # concatenate them and add them to the dataframe
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages_standard)), index=m_d_dependance.columns)
            m_d_dependance.loc[ind] = together
            # move on to the next generation
            ind = ind + 1

    print('A non sis traces finished')

    # the experiment B cell data
    for key in B_non_sis:
        # the dataframe with cycle params
        val = struct.B_dict_non_sis[key]
        val['phi'] = val['generationtime'] * val['growth_length']
        # all the cycle params' means
        trap_averages = np.mean(val)
        # turn them into a standard normal value
        trap_averages_standard = (trap_averages - pop_means) / pop_stds
        # trap_averages = (trap_averages) / pop_stds
        for row1, row2 in zip(range(len(val) - 1), range(1, len(val))):  # row1 is the previous generation of row2, ie. the mom and daughter
            mom = np.array(val.iloc[row1])
            daughter = np.array(val.iloc[row2])
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages)), index=m_d_dependance_units.columns)
            m_d_dependance_units.loc[ind] = together
            # convert them to standard normal values
            mom = (mom - pop_means) / pop_stds
            daughter = (daughter - pop_means) / pop_stds
            # mom = (mom) / pop_stds
            # daughter = (daughter) / pop_stds
            # concatenate them and add them to the dataframe
            together = pd.Series(np.append(mom, np.append(daughter, trap_averages_standard)), index=m_d_dependance.columns)
            m_d_dependance.loc[ind] = together
            # move on to the next generation
            ind = ind + 1

    print('B non sis traces finished')
    
    # SANITY CHECK
    for row_m, row_d in zip(m_d_dependance_units['length_final_m'], m_d_dependance_units['length_final_d']):
        if row_m < 0:
            print('mother length final is negative', row_m)
        if row_d < 0:
            print('daughter length final is negative', row_d)
    

    ''' factors are what we put in to the matrix and targets are what we get out '''
    factors = ['generationtime_m', 'length_birth_m', 'growth_length_m', 'division_ratios__f_n_m', 'length_final_m', 'phi_m']  #
    # 'trap_average_length_final',
    # 'trap_average_division_ratios__f_n', 'trap_average_generationtime', 'trap_average_length_birth',
    #             'trap_average_growth_length'
    targets = ['generationtime_d', 'length_birth_d', 'growth_length_d', 'division_ratios__f_n_d', 'length_final_d', 'phi_d']  #

    ''' put in the struct '''
    struct.factors = factors
    struct.targets = targets

    ''' We leave out final length so that length birth of daughter is not a trivial combination of the division ratio and final length of the mother
    UPDATE: We also leave out the division ratios because we are getting weird results... '''
    X_train, X_test, y_train, y_test = train_test_split(m_d_dependance[factors], m_d_dependance[targets], test_size=0.33, random_state=42)

    struct.X_test = X_test
    struct.X_train = X_train
    struct.y_test = y_test
    struct.y_train = y_train
    struct.m_d_dependance = m_d_dependance
    struct.m_d_dependance_units = m_d_dependance_units

    ''' find the optimum combination of factors for this target '''
    factors_for_iter = ['generationtime_m', 'length_birth_m', 'growth_length_m', 'division_ratios__f_n_m', 'length_final_m', 'phi_m']

    ''' put in the struct '''
    struct.factors_for_iter = factors_for_iter

    pickle_out = open(pickle_name + ".pickle", "wb")
    pickle.dump(struct, pickle_out)
    pickle_out.close()
    
    
def RefiningData(pickle_name, new_data):

    print('new_data=', new_data)

    # For Mac
    infiles_sisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/SISTERS/*.xls')
    infiles_nonsisters = glob.glob(r'/Users/alestawsky/PycharmProjects/untitled/NONSISTERS/*.xls')

    metadatastruct_protein = ssc.CycleData(infiles_sisters=infiles_sisters, infiles_nonsisters=infiles_nonsisters, discretize_by='length',
                                           updated=new_data)

    print(type(metadatastruct_protein))

    pickle_out = open(pickle_name+".pickle", "wb")
    pickle.dump(metadatastruct_protein, pickle_out)
    pickle_out.close()
    print('imported sis/nonsis/both classes')

    # Import the Refined Data
    pickle_in = open(pickle_name+".pickle", "rb")
    metadatastruct_protein = pickle.load(pickle_in)
    pickle_in.close()

    # for Sisters and nonsisters
    # Calculate the betas
    sis_phi_star_A_array, sis_phi_star_B_array, sis_beta_A_array, sis_beta_B_array, sis_pooled_data_dict = \
        CALCULATETHEBETAS1.main(ensemble='Sisters only', graph=False, metadatastruct=metadatastruct_protein, pooled=True)

    # CALCULATE THE ACF OF TRAJECTORY
    sis_acfA_array = np.array([])
    sis_acfB_array = np.array([])
    print(len(metadatastruct_protein.Sisters))
    for dataID in np.arange(len(metadatastruct_protein.Sisters), dtype=int):
        acfA, acfB = metadatastruct_protein.Sisters.AutocorrelationFFT(dataID, normalize=True, maxlen=len(metadatastruct_protein.
                                                                                                          Sisters),
                                                                       enforcelen=False)
        sis_acfA_array = np.append(sis_acfA_array, acfA)
        sis_acfB_array = np.append(sis_acfB_array, acfB)

    # add them as attributes
    metadatastruct_protein.sis_phi_star_A_array = sis_phi_star_A_array
    metadatastruct_protein.sis_phi_star_B_array = sis_phi_star_B_array
    metadatastruct_protein.sis_beta_A_array = sis_beta_A_array
    metadatastruct_protein.sis_beta_B_array = sis_beta_B_array
    metadatastruct_protein.sis_acfA_array = sis_acfA_array
    metadatastruct_protein.sis_acfB_array = sis_acfB_array
    metadatastruct_protein.sis_pooled_data_dict = sis_pooled_data_dict

    # Calculate the betas, maybe plot them
    non_sis_phi_star_A_array, non_sis_phi_star_B_array, non_sis_beta_A_array, non_sis_beta_B_array, non_sis_pooled_data_dict = \
        CALCULATETHEBETAS1.main(ensemble='NonSisters only', graph=False, metadatastruct=metadatastruct_protein, pooled=True)

    non_sis_acfA_array = np.array([])
    non_sis_acfB_array = np.array([])

    print(len(metadatastruct_protein.Nonsisters))
    for dataID in np.arange(len(metadatastruct_protein.Nonsisters), dtype=int):
        acfA1, acfB1 = metadatastruct_protein.Nonsisters.AutocorrelationFFT(dataID, normalize=True,
                                                                            maxlen=len(metadatastruct_protein.Nonsisters), enforcelen=False)
        non_sis_acfA_array = np.append(non_sis_acfA_array, acfA1)
        non_sis_acfB_array = np.append(non_sis_acfB_array, acfB1)

    # add them as attributes
    metadatastruct_protein.non_sis_phi_star_A_array = non_sis_phi_star_A_array
    metadatastruct_protein.non_sis_phi_star_B_array = non_sis_phi_star_B_array
    metadatastruct_protein.non_sis_beta_A_array = non_sis_beta_A_array
    metadatastruct_protein.non_sis_beta_B_array = non_sis_beta_B_array
    metadatastruct_protein.non_sis_acfA_array = non_sis_acfA_array
    metadatastruct_protein.non_sis_acfB_array = non_sis_acfB_array
    metadatastruct_protein.non_sis_pooled_data_dict = non_sis_pooled_data_dict

    # Calculate the betas, maybe plot them
    both_phi_star_A_array, both_phi_star_B_array, both_beta_A_array, both_beta_B_array, both_pooled_data_dict = \
        CALCULATETHEBETAS1.main(ensemble='Both', graph=False, metadatastruct=metadatastruct_protein, pooled=True)

    both_acfA_array = np.array([])
    both_acfB_array = np.array([])

    for trajA, trajB in zip(metadatastruct_protein.A_dict_both.values(), metadatastruct_protein.B_dict_both.values()):
        acfA2, acfB2 = metadatastruct_protein.AutocorrelationFFT1(trajA, trajB, normalize=True, maxlen=len(metadatastruct_protein.A_dict_both),
                                                                  enforcelen=False)
        both_acfA_array = np.append(both_acfA_array, acfA2)
        both_acfB_array = np.append(both_acfB_array, acfB2)

    # add them as attributes
    metadatastruct_protein.both_phi_star_A_array = both_phi_star_A_array
    metadatastruct_protein.both_phi_star_B_array = both_phi_star_B_array
    metadatastruct_protein.both_beta_A_array = both_beta_A_array
    metadatastruct_protein.both_beta_B_array = both_beta_B_array
    metadatastruct_protein.both_acfA_array = both_acfA_array
    metadatastruct_protein.both_acfB_array = both_acfB_array
    metadatastruct_protein.both_pooled_data_dict = both_pooled_data_dict

    # # # # # # # # # # # # #  # # #  # # # #  # # # # # #  # # # #  #

    # FOR THE "TRUSTWORTHY (HOPEFULLY)" BETAS
    s_phi_star_A_array, s_phi_star_B_array, s_beta_A_array, s_beta_B_array, s_pooled_data_dict = \
        CALCULATETHEBETAS.main(ensemble='Sisters only', graph=False, metadatastruct=metadatastruct_protein, pooled=True)

    # CALCULATE THE ACF OF TRAJECTORY
    s_acfA_array = np.array([])
    s_acfB_array = np.array([])
    print(len(metadatastruct_protein.Sisters))
    for dataID in np.arange(len(metadatastruct_protein.Sisters), dtype=int):
        acfA, acfB = metadatastruct_protein.Sisters.AutocorrelationFFT(dataID, normalize=True, maxlen=len(metadatastruct_protein.
                                                                                                          Sisters),
                                                                       enforcelen=False)
        s_acfA_array = np.append(s_acfA_array, acfA)
        s_acfB_array = np.append(s_acfB_array, acfB)

    # add them as attributes
    metadatastruct_protein.s_phi_star_A_array = s_phi_star_A_array
    metadatastruct_protein.s_phi_star_B_array = s_phi_star_B_array
    metadatastruct_protein.s_beta_A_array = s_beta_A_array
    metadatastruct_protein.s_beta_B_array = s_beta_B_array
    metadatastruct_protein.s_acfA_array = s_acfA_array
    metadatastruct_protein.s_acfB_array = s_acfB_array
    metadatastruct_protein.s_pooled_data_dict = s_pooled_data_dict

    # Calculate the betas, maybe plot them
    n_phi_star_A_array, n_phi_star_B_array, n_beta_A_array, n_beta_B_array, n_pooled_data_dict = \
        CALCULATETHEBETAS.main(ensemble='NonSisters only', graph=False, metadatastruct=metadatastruct_protein, pooled=True)

    n_acfA_array = np.array([])
    n_acfB_array = np.array([])

    print(len(metadatastruct_protein.Nonsisters))
    for dataID in np.arange(len(metadatastruct_protein.Nonsisters), dtype=int):
        acfA1, acfB1 = metadatastruct_protein.Nonsisters.AutocorrelationFFT(dataID, normalize=True,
                                                                            maxlen=len(metadatastruct_protein.Nonsisters),
                                                                            enforcelen=False)
        n_acfA_array = np.append(n_acfA_array, acfA1)
        n_acfB_array = np.append(n_acfB_array, acfB1)

    # add them as attributes
    metadatastruct_protein.n_phi_star_A_array = n_phi_star_A_array
    metadatastruct_protein.n_phi_star_B_array = n_phi_star_B_array
    metadatastruct_protein.n_beta_A_array = n_beta_A_array
    metadatastruct_protein.n_beta_B_array = n_beta_B_array
    metadatastruct_protein.n_acfA_array = n_acfA_array
    metadatastruct_protein.n_acfB_array = n_acfB_array
    metadatastruct_protein.n_pooled_data_dict = n_pooled_data_dict

    # Calculate the betas, maybe plot them
    b_phi_star_A_array, b_phi_star_B_array, b_beta_A_array, b_beta_B_array, b_pooled_data_dict = \
        CALCULATETHEBETAS.main(ensemble='Both', graph=False, metadatastruct=metadatastruct_protein, pooled=True)

    b_acfA_array = np.array([])
    b_acfB_array = np.array([])

    for trajA, trajB in zip(metadatastruct_protein.A_dict_both.values(), metadatastruct_protein.B_dict_both.values()):
        acfA2, acfB2 = metadatastruct_protein.AutocorrelationFFT1(trajA, trajB, normalize=True,
                                                                  maxlen=len(metadatastruct_protein.A_dict_both),
                                                                  enforcelen=False)
        b_acfA_array = np.append(b_acfA_array, acfA2)
        b_acfB_array = np.append(b_acfB_array, acfB2)

    # add them as attributes
    metadatastruct_protein.b_phi_star_A_array = b_phi_star_A_array
    metadatastruct_protein.b_phi_star_B_array = b_phi_star_B_array
    metadatastruct_protein.b_beta_A_array = b_beta_A_array
    metadatastruct_protein.b_beta_B_array = b_beta_B_array
    metadatastruct_protein.b_acfA_array = b_acfA_array
    metadatastruct_protein.b_acfB_array = b_acfB_array
    metadatastruct_protein.b_pooled_data_dict = b_pooled_data_dict

    InitiationOnlyOnce(metadatastruct_protein, pickle_name=pickle_name)

    pickle_out = open(pickle_name+".pickle", "wb")
    pickle.dump(metadatastruct_protein, pickle_out)
    pickle_out.close()

    print('imported sis')

    print(len(metadatastruct_protein.A_dict_both), len(metadatastruct_protein.B_dict_both))
    

def main():
    RefiningData('metastructdata', new_data=True)
    RefiningData('metastructdata_old', new_data=False)


if __name__ == '__main__':
    main()
