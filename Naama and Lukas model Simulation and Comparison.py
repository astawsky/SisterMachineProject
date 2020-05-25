import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from correlation_dataframe_and_heatmap import output_heatmap
from correlation_dataframe_and_heatmap import correlation_dataframe
import pickle
import scipy.stats as stats

'''
Graph how different the intercepts are from ln(2) and maybe using the <phi> = ln(2) is a little bit irrealistic
'''
def graph_different_betas(struct, pop_means, do_I_print = False, do_I_pool_both_traces = False):
    Beta_df = pd.DataFrame(columns=['A', 'B'])
    for trap_number in np.arange(len(struct.Sisters)):

        # Pre-processing, name the traces, normalize length at birth, make fold growth column
        trajA = struct.A_dict_sis['Sister_Trace_A_' + str(trap_number)]
        trajB = struct.B_dict_sis['Sister_Trace_B_' + str(trap_number)]
        trajA['fold_growth'] = trajA['generationtime'] * trajA['growth_length']
        trajB['fold_growth'] = trajB['generationtime'] * trajB['growth_length']
        trajA['length_birth'] = np.log(trajA['length_birth'] / pop_means[1])
        trajB['length_birth'] = np.log(trajB['length_birth'] / pop_means[1])

        ''' Here we use both traces for the BETA linear regression '''
        if do_I_pool_both_traces == True:
            # put A and B trace together
            combined_length_trap_array = pd.concat([trajA['length_birth'], trajB['length_birth']])
            combined_fold_growth_trap_array = pd.concat([trajA['fold_growth'], trajB['fold_growth']])

            reg = LR().fit(np.array(combined_length_trap_array).reshape(-1, 1), np.array(combined_fold_growth_trap_array).reshape(-1, 1))
            if do_I_print == True:
                print('the x-star in the normalization of length', pop_means[1])
                print('slope', reg.coef_)
                print('intercept', reg.intercept_)

            # now plot the trace
            plt.axvline(x=0, color='black')
            plt.axhline(y=np.log(2), color='black')
            plt.plot(np.linspace(min(combined_length_trap_array), max(combined_length_trap_array), num=50),
                     reg.intercept_ + reg.coef_[0] * np.linspace(min(
                         combined_length_trap_array), max(combined_length_trap_array), num=50), color='red', label='with fitted intercept')
            plt.plot(np.linspace(min(combined_length_trap_array), max(combined_length_trap_array), num=50),
                     np.log(2) + reg.coef_[0] * np.linspace(min(combined_length_trap_array), max(combined_length_trap_array), num=50),
                     color='green', label='with log(2) intercept')
            plt.scatter(combined_length_trap_array, combined_fold_growth_trap_array, facecolors='none', edgecolors='blue')
            plt.title('Pooling both Traces')
            plt.xlabel('normalized length at birth')
            plt.ylabel('fold growth')
            plt.legend()
            plt.show()
            plt.close()

        ''' Here we use each trace separately for the BETA linear regression '''
        if do_I_pool_both_traces == False:
            # first do it for A trace
            regA = LR().fit(np.array(trajA['length_birth']).reshape(-1, 1), np.array(trajA['fold_growth']).reshape(-1, 1))
            if do_I_print == True:
                print('the x-star in the normalization of length', pop_means[1])
                print('slope', regA.coef_)
                print('intercept', regA.intercept_)

            # then do it for B trace
            regB = LR().fit(np.array(trajB['length_birth']).reshape(-1, 1), np.array(trajB['fold_growth']).reshape(-1, 1))
            if do_I_print == True:
                print('the x-star in the normalization of length', pop_means[1])
                print('slope', regB.coef_)
                print('intercept', regB.intercept_)

            # save the betas
            Beta_df['A'].loc[trap_number] = regA.coef_
            Beta_df['B'].loc[trap_number] = regB.coef_

            # now plot A trace
            domain = np.linspace(min([min(trajA['length_birth']), min(trajB['length_birth'])]),
                                 max([max(trajA['length_birth']), max(trajB['length_birth'])]), num=50)
            plt.axvline(x=0, color='black')
            plt.axhline(y=np.log(2), color='black')
            plt.plot(domain, regA.intercept_ + regA.coef_[0] * domain, color='red', label='trace A with fitted intercept {}'.format(
                round(regA.intercept_[0], 2)))
            plt.plot(domain, np.log(2) + regA.coef_[0] * domain, color='orange', label='trace A with log(2) intercept')
            plt.scatter(trajA['length_birth'], trajA['fold_growth'], facecolors='none', edgecolors='red', label='_nolegend_')
            # plot B trace
            plt.plot(domain, regB.intercept_ + regB.coef_[0] * domain, color='blue', label='trace B with fitted intercept {}'.format(
                round(regB.intercept_[0], 2)))
            plt.plot(domain, np.log(2) + regB.coef_[0] * domain, color='cyan', label='trace B with log(2) intercept')
            plt.scatter(trajB['length_birth'], trajB['fold_growth'], facecolors='none', edgecolors='blue', label='_nolegend_')
            # meta plotting parameters
            plt.title('Doing each trace separately')
            plt.xlabel('normalized length at birth')
            plt.ylabel('fold growth')
            plt.legend()
            plt.show()
            plt.close()


def elbowGraphOfSyntheticSimulation(column_names, beta, final_cycle_number=30, number_of_traces=100, c_1=5, c_2=5, growth_average=1.32):

    df_repository = []
    new_df = pd.DataFrame(columns=np.array([['Mother ' + col + ' and daughter ' + row for row in column_names] for col in
                                            column_names]).reshape(-1))
    for how_many_cycles in np.arange(2, final_cycle_number):
        print('how many cycles:', how_many_cycles)
        all_traces_df, mother_df, daughter_df = completely_synthetic_simulation(growth_average=growth_average, beta=beta, c_1=c_1, c_2=c_2,
                                                                                number_of_traces=number_of_traces, cycles_per_trace=how_many_cycles)
        m_d_corr_df = correlation_dataframe(mother_df, daughter_df)

        df_repository.append(m_d_corr_df)

    for all_cycle_tries in np.arange(len(df_repository)):
        this_df = df_repository[all_cycle_tries]
        what_to_add = dict()
        for col in m_d_corr_df.columns:
            for row in m_d_corr_df.index:
                what_to_add.update({'Mother ' + col + ' and daughter ' + row: this_df[col].loc[row]})
        new_df = new_df.append(what_to_add, ignore_index=True)

    for col in new_df.columns:
        plt.plot(np.arange(2, final_cycle_number), new_df[col], label=col, marker='.')

    # plt.legend(), because they were too many
    plt.text(0, .7, 'number of traces: '+str(number_of_traces)+'\n beta: '+str(beta)+'\n growth average: '+str(
        growth_average)+'\n c_1, c_2: '+str(c_1)+', '+str(c_2), wrap=True)
    plt.xlabel('the number of cycles per trace')
    plt.ylabel('value of relationship correlation')
    plt.show()
    # plt.savefig('elbow graph for simulations', dpi=300)


'''
This outputs the correlations of the dset data and simulated data for only NORMALIZED LENGTH AT BIRTH and CYCLE DURATION
The GROWTH RATE and DIVISION FRACTION are fixed depending on the scheme below:
For A and B we can choose from the following options:
    betaA, betaB, beta_trap, betaPopulation
    A_time_avg, B_time_avg, trap_time_avg, pop_time_avg
    A_growth_avg, B_growth_avg, trap_growth_avg, pop_growth_avg
    interceptA, interceptB, intercept_trap, interceptPopulation
'''
def SimulationDataCorrelation(dset, beta_A_number, beta_B_number, time_A_number, time_B_number, growth_A_number, growth_B_number,
                                intercept_A_number, intercept_B_number):
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # simple renaming
    factors = struct.factors
    targets = struct.targets
    X_test = struct.X_test
    X_train = struct.X_train
    y_test = struct.y_test
    y_train = struct.y_train
    m_d_dependance_units = struct.m_d_dependance_units
    cols = struct.cols
    pop_means = struct.pop_means
    pop_stds = struct.pop_stds

    A_correlations_array = []
    B_correlations_array = []
    A_correlations_pval_array = []
    B_correlations_pval_array = []
    pop_time_avg = pop_means[0]
    pop_growth_avg = pop_means[2]
    regPopulation = LR().fit(np.array(m_d_dependance_units['length_birth_m']).reshape(-1, 1),
                             np.array(m_d_dependance_units['phi_m']).reshape(-1, 1))
    betaPopulation = regPopulation.coef_[0]
    interceptPopulation = regPopulation.intercept_

    # deciding what to use depending on the dataset:
    if dset == 'Sisters':
        raw = struct.Sisters
        cycleA = struct.A_dict_sis
        cycleB = struct.B_dict_sis
    elif dset == 'Non-Sisters':
        raw = struct.Nonsisters
        cycleA = struct.A_dict_non_sis
        cycleB = struct.B_dict_non_sis
    elif dset == 'Control':
        raw = struct.Control[0]  # just to get the amount of traces
        cycleA = struct.A_dict_both
        cycleB = struct.B_dict_both
    else:
        IOError("Choose a suitable dset: 'Sisters', 'Non-Sisters' or 'Control'")

    for trap_number in np.arange(len(raw)):
        # Pre-processing, name the traces, normalize length at birth, make fold growth column
        trajA = cycleA['Sister_Trace_A_' + str(trap_number)]
        trajB = cycleB['Sister_Trace_B_' + str(trap_number)]
        trajA['fold_growth'] = trajA['generationtime'] * trajA['growth_length']
        trajB['fold_growth'] = trajB['generationtime'] * trajB['growth_length']
        trajA['length_birth'] = np.log(trajA['length_birth'] / pop_means[1])
        trajB['length_birth'] = np.log(trajB['length_birth'] / pop_means[1])

        # Calculating the time average, population is calculated outside of the loop
        A_time_avg = np.mean(trajA['generationtime'])
        B_time_avg = np.mean(trajB['generationtime'])
        trap_time_avg = np.mean(pd.concat([trajA['generationtime'], trajB['generationtime']]))

        # Calculating the growth rate average, population is calculated outside of the loop
        A_growth_avg = np.mean(trajA['growth_length'])
        B_growth_avg = np.mean(trajB['growth_length'])
        trap_growth_avg = np.mean(pd.concat([trajA['growth_length'], trajB['growth_length']]))

        # Calculating the beta and intercept (not always ln(2)), population is calculated outside of the loop
        regA = LR().fit(np.array(trajA['length_birth']).reshape(-1, 1), np.array(trajA['fold_growth']).reshape(-1, 1))
        betaA = regA.coef_[0]
        interceptA = regA.intercept_
        regB = LR().fit(np.array(trajB['length_birth']).reshape(-1, 1), np.array(trajB['fold_growth']).reshape(-1, 1))
        betaB = regB.coef_[0]
        interceptB = regB.intercept_
        pooled_length = np.array(pd.concat([trajA['length_birth'], trajB['length_birth']]))
        pooled_fold_growth = np.array(pd.concat([trajA['fold_growth'], trajB['fold_growth']]))
        reg_trap = LR().fit(pooled_length.reshape(-1, 1), pooled_fold_growth.reshape(-1, 1))
        beta_trap = reg_trap.coef_[0]
        intercept_trap = reg_trap.intercept_

        # Choose which ones we will use
        chosen_beta_A = [betaA, betaB, beta_trap, betaPopulation][beta_A_number]
        chosen_beta_B = [betaA, betaB, beta_trap, betaPopulation][beta_B_number]
        chosen_time_avg_A = [A_time_avg, B_time_avg, trap_time_avg, pop_time_avg][time_A_number]
        chosen_time_avg_B = [A_time_avg, B_time_avg, trap_time_avg, pop_time_avg][time_B_number]
        chosen_growth_avg_A = [A_growth_avg, B_growth_avg, trap_growth_avg, pop_growth_avg][growth_A_number]
        chosen_growth_avg_B = [A_growth_avg, B_growth_avg, trap_growth_avg, pop_growth_avg][growth_B_number]
        chosen_intercept_A = [interceptA, interceptB, intercept_trap, interceptPopulation][intercept_A_number]  # just for knowledge
        chosen_intercept_B = [interceptA, interceptB, intercept_trap, interceptPopulation][intercept_B_number]  # just for knowledge

        # defining the model matrices
        model_matrix_A = np.array([[1, chosen_growth_avg_A], [-chosen_beta_A[0] / chosen_growth_avg_A, -chosen_beta_A[0]]])
        model_matrix_B = np.array([[1, chosen_growth_avg_B], [-chosen_beta_B[0] / chosen_growth_avg_B, -chosen_beta_B[0]]])

        # what we will feed into the model
        samples_A = np.array([[length, dT] for length, dT in zip(trajA['length_birth'][:-1], trajA['generationtime'][:-1] - chosen_time_avg_A)])
        samples_B = np.array([[length, dT] for length, dT in zip(trajB['length_birth'][:-1], trajB['generationtime'][:-1] - chosen_time_avg_B)])

        # what the model will hopefully predict
        predict_A = np.array([[length, dT] for length, dT in zip(trajA['length_birth'][1:], trajA['generationtime'][1:] - chosen_time_avg_A)])
        predict_B = np.array([[length, dT] for length, dT in zip(trajB['length_birth'][1:], trajB['generationtime'][1:] - chosen_time_avg_B)])

        # simulated variables
        A_simulated = np.array([np.dot(model_matrix_A, samples_A[index].T) for index in range(samples_A.shape[0])])
        B_simulated = np.array([np.dot(model_matrix_B, samples_B[index].T) for index in range(samples_B.shape[0])])

        # actual correlation
        A_correlation_between_prediction_and_data = stats.pearsonr(A_simulated[:, 0], predict_A[:, 0])
        B_correlation_between_prediction_and_data = stats.pearsonr(B_simulated[:, 0], predict_B[:, 0])

        # put it into the correlation arrays and the pvalue arrays since we might have very small number of generations ie points
        A_correlations_array.append(A_correlation_between_prediction_and_data[0])
        B_correlations_array.append(B_correlation_between_prediction_and_data[0])
        A_correlations_pval_array.append(A_correlation_between_prediction_and_data[1])
        B_correlations_pval_array.append(B_correlation_between_prediction_and_data[1])

    # make them numpy arrays and flatten them
    A_correlations_array = np.array(A_correlations_array).reshape(-1)
    B_correlations_array = np.array(B_correlations_array).reshape(-1)
    A_correlations_pval_array = np.array(A_correlations_pval_array).reshape(-1)
    B_correlations_pval_array = np.array(B_correlations_pval_array).reshape(-1)

    return A_correlations_array, B_correlations_array, A_correlations_pval_array, B_correlations_pval_array


'''

'''
def Simulation(dset, beta_A_number, beta_B_number, time_A_number, time_B_number, growth_A_number, growth_B_number,
                                intercept_A_number, intercept_B_number):
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # simple renaming
    factors = struct.factors
    targets = struct.targets
    X_test = struct.X_test
    X_train = struct.X_train
    y_test = struct.y_test
    y_train = struct.y_train
    m_d_dependance_units = struct.m_d_dependance_units
    cols = struct.cols
    pop_means = struct.pop_means
    pop_stds = struct.pop_stds

    # defining dataframes for simulated data
    column_names = ['generationtime', 'log-normalized_length_birth', 'growth_length', 'division_ratios__f_n_m', 'log-normalized_length_final',
                    'phi']
    A_simulated_traces = pd.DataFrame(columns=column_names)
    B_simulated_traces = pd.DataFrame(columns=column_names)
    A_mother_simulated_traces = pd.DataFrame(columns=column_names)
    B_mother_simulated_traces = pd.DataFrame(columns=column_names)
    A_daughter_simulated_traces = pd.DataFrame(columns=column_names)
    B_daughter_simulated_traces = pd.DataFrame(columns=column_names)

    pop_time_avg = pop_means[0]
    pop_growth_avg = pop_means[2]
    regPopulation = LR().fit(np.array(m_d_dependance_units['length_birth_m']).reshape(-1, 1),
                             np.array(m_d_dependance_units['phi_m']).reshape(-1, 1))
    betaPopulation = regPopulation.coef_[0]
    interceptPopulation = regPopulation.intercept_

    # deciding what to use depending on the dataset:
    if dset == 'Sisters':
        raw = struct.Sisters
        cycleA = struct.A_dict_sis
        cycleB = struct.B_dict_sis
    elif dset == 'Non-Sisters':
        raw = struct.Nonsisters
        cycleA = struct.A_dict_non_sis
        cycleB = struct.B_dict_non_sis
    elif dset == 'Control':
        raw = struct.Control[0]  # just to get the amount of traces
        cycleA = struct.A_dict_both
        cycleB = struct.B_dict_both
    else:
        IOError("Choose a suitable dset: 'Sisters', 'Non-Sisters' or 'Control'")

    for trap_number in np.arange(len(raw)):
        
        # Pre-processing, name the traces, normalize length at birth, make fold growth column
        trajA = cycleA['Sister_Trace_A_' + str(trap_number)]
        trajB = cycleB['Sister_Trace_B_' + str(trap_number)]
        trajA['fold_growth'] = trajA['generationtime'] * trajA['growth_length']
        trajB['fold_growth'] = trajB['generationtime'] * trajB['growth_length']
        trajA['length_birth'] = np.log(trajA['length_birth'] / pop_means[1])
        trajB['length_birth'] = np.log(trajB['length_birth'] / pop_means[1])

        # Calculating the time average, population is calculated outside of the loop
        A_time_avg = np.mean(trajA['generationtime'])
        B_time_avg = np.mean(trajB['generationtime'])
        trap_time_avg = np.mean(pd.concat([trajA['generationtime'], trajB['generationtime']]))

        # Calculating the growth rate average, population is calculated outside of the loop
        A_growth_avg = np.mean(trajA['growth_length'])
        B_growth_avg = np.mean(trajB['growth_length'])
        trap_growth_avg = np.mean(pd.concat([trajA['growth_length'], trajB['growth_length']]))

        # Calculating the beta and intercept (not always ln(2)), population is calculated outside of the loop
        regA = LR().fit(np.array(trajA['length_birth']).reshape(-1, 1), np.array(trajA['fold_growth']).reshape(-1, 1))
        betaA = regA.coef_[0]
        interceptA = regA.intercept_
        regB = LR().fit(np.array(trajB['length_birth']).reshape(-1, 1), np.array(trajB['fold_growth']).reshape(-1, 1))
        betaB = regB.coef_[0]
        interceptB = regB.intercept_
        pooled_length = np.array(pd.concat([trajA['length_birth'], trajB['length_birth']]))
        pooled_fold_growth = np.array(pd.concat([trajA['fold_growth'], trajB['fold_growth']]))
        reg_trap = LR().fit(pooled_length.reshape(-1, 1), pooled_fold_growth.reshape(-1, 1))
        beta_trap = reg_trap.coef_[0]
        intercept_trap = reg_trap.intercept_

        # Choose which ones we will use
        chosen_beta_A = [betaA, betaB, beta_trap, betaPopulation][beta_A_number]
        chosen_beta_B = [betaA, betaB, beta_trap, betaPopulation][beta_B_number]
        chosen_time_avg_A = [A_time_avg, B_time_avg, trap_time_avg, pop_time_avg][time_A_number]
        chosen_time_avg_B = [A_time_avg, B_time_avg, trap_time_avg, pop_time_avg][time_B_number]
        chosen_growth_avg_A = [A_growth_avg, B_growth_avg, trap_growth_avg, pop_growth_avg][growth_A_number]
        chosen_growth_avg_B = [A_growth_avg, B_growth_avg, trap_growth_avg, pop_growth_avg][growth_B_number]
        chosen_intercept_A = [interceptA, interceptB, intercept_trap, interceptPopulation][intercept_A_number]  # just for knowledge
        chosen_intercept_B = [interceptA, interceptB, intercept_trap, interceptPopulation][intercept_B_number]  # just for knowledge

        # defining the model matrices
        model_matrix_A = np.array([[1, chosen_growth_avg_A], [-chosen_beta_A[0] / chosen_growth_avg_A, -chosen_beta_A[0]]])
        model_matrix_B = np.array([[1, chosen_growth_avg_B], [-chosen_beta_B[0] / chosen_growth_avg_B, -chosen_beta_B[0]]])

        # what we will feed into the model
        samples_A = np.array([[length, dT] for length, dT in zip(trajA['length_birth'][:-1], trajA['generationtime'][:-1] - chosen_time_avg_A)])
        samples_B = np.array([[length, dT] for length, dT in zip(trajB['length_birth'][:-1], trajB['generationtime'][:-1] - chosen_time_avg_B)])

        # # what the model will hopefully predict
        # predict_A = np.array([[length, dT] for length, dT in zip(trajA['length_birth'][1:], trajA['generationtime'][1:] - chosen_time_avg_A)])
        # predict_B = np.array([[length, dT] for length, dT in zip(trajB['length_birth'][1:], trajB['generationtime'][1:] - chosen_time_avg_B)])

        # simulated variables
        A_simulated = np.array([np.dot(model_matrix_A, samples_A[index].T) for index in range(samples_A.shape[0])])
        B_simulated = np.array([np.dot(model_matrix_B, samples_B[index].T) for index in range(samples_B.shape[0])])

        # The simulated traces, 0.5 for f because that's what its fixed to
        data_A = [ A_simulated[:,1], A_simulated[:,0], chosen_growth_avg_A * np.ones_like(A_simulated[:,1]), 0.5 * np.ones_like(A_simulated[:,1]),
                    A_simulated[:,0] + chosen_growth_avg_A * A_simulated[:,1], chosen_growth_avg_A * A_simulated[:,1] ]

        data_B = [B_simulated[:, 1], B_simulated[:, 0], chosen_growth_avg_B * np.ones_like(B_simulated[:, 1]), 0.5 * np.ones_like(B_simulated[:, 1]),
                   B_simulated[:, 0] + chosen_growth_avg_B * B_simulated[:, 1], chosen_growth_avg_B * B_simulated[:, 1]]

        mother_data_A = [A_simulated[:, 1][:-1], A_simulated[:, 0][:-1], chosen_growth_avg_A * np.ones_like(A_simulated[:, 1])[:-1], 0.5 *
                         np.ones_like(A_simulated[:, 1])[:-1], A_simulated[:, 0][:-1] + chosen_growth_avg_A * A_simulated[:, 1][:-1],
                         chosen_growth_avg_A * A_simulated[:,1][:-1]]

        mother_data_B = [B_simulated[:, 1][:-1], B_simulated[:, 0][:-1], chosen_growth_avg_B * np.ones_like(B_simulated[:, 1])[:-1], 0.5 *
                         np.ones_like(B_simulated[:, 1])[:-1], B_simulated[:, 0][:-1] + chosen_growth_avg_B * B_simulated[:, 1][:-1],
                         chosen_growth_avg_B * B_simulated[:, 1][:-1]]

        daughter_data_A = [A_simulated[:, 1][1:], A_simulated[:, 0][1:], chosen_growth_avg_A * np.ones_like(A_simulated[:, 1][1:]), 0.5 *
                         np.ones_like(A_simulated[:, 1][1:]), A_simulated[:, 0][1:] + chosen_growth_avg_A * A_simulated[:, 1][1:],
                         chosen_growth_avg_A * A_simulated[:, 1][1:]]

        daughter_data_B = [B_simulated[:, 1][1:], B_simulated[:, 0][1:], chosen_growth_avg_B * np.ones_like(B_simulated[:, 1])[1:], 0.5 *
                         np.ones_like(B_simulated[:, 1])[1:], B_simulated[:, 0][1:] + chosen_growth_avg_B * B_simulated[:, 1][1:],
                         chosen_growth_avg_B * B_simulated[:, 1][1:]]


        # Whole-Trace dataframes
        trap_df_A = pd.DataFrame(data=dict(zip(column_names, data_A)))
        trap_df_B = pd.DataFrame(data=dict(zip(column_names, data_B)))

        # Mother-Daughter
        mother_trap_df_A = pd.DataFrame(data=dict(zip(column_names, mother_data_A)))
        mother_trap_df_B = pd.DataFrame(data=dict(zip(column_names, mother_data_B)))
        daughter_trap_df_A = pd.DataFrame(data=dict(zip(column_names, daughter_data_A)))
        daughter_trap_df_B = pd.DataFrame(data=dict(zip(column_names, daughter_data_B)))

        A_simulated_traces = A_simulated_traces.append(trap_df_A)
        B_simulated_traces = B_simulated_traces.append(trap_df_B)

        A_mother_simulated_traces = A_mother_simulated_traces.append(mother_trap_df_A)
        B_mother_simulated_traces = B_mother_simulated_traces.append(mother_trap_df_B)
        A_daughter_simulated_traces = A_daughter_simulated_traces.append(daughter_trap_df_A)
        B_daughter_simulated_traces = B_daughter_simulated_traces.append(daughter_trap_df_B)

    return A_simulated_traces, B_simulated_traces, A_mother_simulated_traces, B_mother_simulated_traces, A_daughter_simulated_traces,\
           B_daughter_simulated_traces


'''
c_1 and c_2 are the coefficients of noise terms, y and tau
the number of traces we will manufacture
how many cycles we will record per trace
'''
def completely_synthetic_simulation(growth_average, beta, c_1=0.2, c_2=0.2, number_of_traces = 800, cycles_per_trace = 50, column_names =
        ['generationtime', 'log-normalized_length_birth', 'growth_length', 'division_ratios__f_n', 'log-normalized_length_final', 'phi']):
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # simple renaming
    m_d_dependance_units = struct.m_d_dependance_units

    # the model matrix with chosen growth rate and beta
    model_matrix = np.array([[1, growth_average], [-beta / growth_average, -beta]])

    # the random initial variables from which the 50 traces will stem from
    mean_of_length = np.mean(np.log(m_d_dependance_units['length_birth_m'] / struct.x_avg))
    std_of_length = np.std(np.log(m_d_dependance_units['length_birth_m'] / struct.x_avg))
    initial_normalized_lengths = np.random.normal(loc=mean_of_length, scale=std_of_length, size=number_of_traces)
    # notice that now we are subtracting the global generationtime average
    initial_deviation_of_gentime_from_mean = np.random.normal(loc=0, scale=np.std(m_d_dependance_units['generationtime_m']), size=number_of_traces)

    # print(initial_normalized_lengths, initial_deviation_of_gentime_from_mean)

    # Going to hold all the synthetic trajectories
    all_traces_df = pd.DataFrame(columns=column_names)
    for trace_num in range(number_of_traces):

        # using the previous generation to predict the next, previous_generation[0] = y_n and previous_generation[1] = delta_tau_n
        previous_generation = np.array([initial_normalized_lengths[trace_num], initial_deviation_of_gentime_from_mean[trace_num]])

        # print('previous generation\n', previous_generation)

        for cycle_num in range(cycles_per_trace):

            # Convert delta_tau_n to tau_n so as to compare to the data, then the log normal length birth y_n, growth average, division ratio,
            # then the log normal final length based on the exponential model, finally the fold growth. Essentially looks like:
            # ['generationtime(added the mean)', 'log-normalized_length_birth', 'growth_length', 'division_ratios__f_n_m',
            # 'log-normalized_length_final', 'phi']
            tau_n = previous_generation[1] + np.mean(m_d_dependance_units['generationtime_m'])
            y_n = previous_generation[0]
            phi_n = growth_average * tau_n
            log_normal_final_length = previous_generation[0] + phi_n

            # Arrange it into a pandas series/array
            what_to_append = pd.Series([tau_n, y_n, growth_average, 0.5, log_normal_final_length, phi_n])
            # Define their order with the function parameter column_names
            all_traces_df = all_traces_df.append(dict(zip(column_names, what_to_append)), ignore_index=True)

            # print('what_to_append\n', what_to_append)

            # standard normal noise with coefficients named c_1, c_2
            noise = np.array([c_1 * np.random.normal(0, 1, 1), c_2 * np.random.normal(0, 1, 1)])

            # print('noise\n', noise)

            # Get the next generation which will become the previous generation so that's what I name it, recursive programming
            previous_generation = np.dot(model_matrix, previous_generation.T) + noise.T

            # this is because, for some reason, the array from the previous action gives the vector inside another, useless array and putting [0]
            # takes it out of this useless array and we get the vector
            previous_generation = previous_generation[0]

    # mother and daughter dataframes
    daughter_df = pd.DataFrame(columns=all_traces_df.columns)
    mother_df = pd.DataFrame(columns=all_traces_df.columns)
    
    # if the daughter index is a multiple of how many cycles per trace then they are not really mother-daughter and we can't add them to the
    # dataframes
    for mother_index in range(len(all_traces_df) - 1):
        if np.mod(mother_index+1, cycles_per_trace) != 0:
            daughter_df = daughter_df.append(all_traces_df.loc[mother_index+1])
            mother_df = mother_df.append(all_traces_df.loc[mother_index])

    return all_traces_df, mother_df, daughter_df


def look_at_trajectories(all_traces_df, cycles_per_trace):
    for trace in np.arange(10):
        starting_point = trace * cycles_per_trace
        length = np.array([all_traces_df['log-normalized_length_birth'].iloc[starting_point + cycle] for cycle in np.arange(cycles_per_trace)])
        time = np.array([all_traces_df['generationtime'].iloc[starting_point + cycle] for cycle in np.arange(cycles_per_trace)])
        plt.plot(np.arange(cycles_per_trace), time, label='generationtime', marker='.')
        plt.plot(np.arange(cycles_per_trace), length, label='length at birth', marker='.')
        plt.legend()
        plt.title('are they oscillating?')
        plt.xlabel('generations')
        plt.show()
        plt.close()


def run_the_simulation(pickle_name, number_of_traces, cycles_per_trace, c_1, c_2, growth_average,
                       positive_beta):

    # Import the Refined Data
    pickle_in = open(pickle_name+".pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    print('struct.Synthetic_Simulations_Archive type', type(struct.Synthetic_Simulations_Archive), struct.Synthetic_Simulations_Archive)

    # simple renaming
    factors = struct.factors
    targets = struct.targets
    X_test = struct.X_test
    X_train = struct.X_train
    y_test = struct.y_test
    y_train = struct.y_train
    m_d_dependance_units = struct.m_d_dependance_units
    cols = struct.cols
    pop_means = struct.pop_means
    pop_stds = struct.pop_stds

    regPopulation = LR().fit(np.array(np.log(m_d_dependance_units['length_birth_m']/np.mean(m_d_dependance_units['length_birth_m']))).reshape(-1, 1),
                             np.array(m_d_dependance_units['generationtime_m']-np.mean(m_d_dependance_units['generationtime_m'])).reshape(-1, 1))
    betaPopulation = regPopulation.coef_[0]
    interceptPopulation = regPopulation.intercept_

    column_names = ['generationtime', 'log-normalized_length_birth', 'growth_length', 'division_ratios__f_n', 'log-normalized_length_final',
                    'phi']

    # elbow graph
    # elbowGraphOfSyntheticSimulation(column_names=column_names, beta=betaPopulation, final_cycle_number=20, number_of_traces=100, c_1=5, c_2=5,
    #                                 growth_average=1.32)
    # exit()

    # number_of_traces = 20
    # cycles_per_trace = 20
    # c_1 = 50
    # c_2 = 50
    # growth_average = 1.32
    if positive_beta is None:
        positive_beta = -betaPopulation

    print(positive_beta)
    all_traces_df, mother_df, daughter_df = completely_synthetic_simulation(growth_average=growth_average, beta=positive_beta, c_1=c_1, c_2=c_2,
                                            number_of_traces=number_of_traces, cycles_per_trace=cycles_per_trace, column_names = column_names)

    # look_at_trajectories(all_traces_df, cycles_per_trace=cycles_per_trace)

    # print([[col, np.mean(all_traces_df[col])] for col in all_traces_df.columns])
    # exit()

    # print(np.array([[all_traces_df['log-normalized_length_birth'].loc[10*ind] for l in 100] for ind in 100]))
    #
    # plt.scatter(np.array([[all_traces_df['log-normalized_length_birth'].loc[10*ind] for l in 100] for ind in 100]), all_traces_df['phi'])
    # plt.show()

    # print(np.mean(all_traces_df['phi']))
    # exit()

    # Mother/Daughter, Sister dataset
    m_d_corr_df = correlation_dataframe(mother_df, daughter_df)

    # since the division ratios and growth length are fixed, to make it look better we take it out of the heatmap
    columns_to_show = ['generationtime', 'log-normalized_length_birth', 'log-normalized_length_final', 'phi']
    columns_to_show_m = [col + '_m' for col in columns_to_show]
    columns_to_show_d = [col + '_d' for col in columns_to_show]

    key = str(number_of_traces) + ', ' + str(cycles_per_trace) + ', ' + str(c_1) + ', ' + str(c_1)
    key = str(key)

    # renaming
    Synthetic_Simulations_Archive = struct.Synthetic_Simulations_Archive

    # If Synthetic_Simulations_Archive was never created
    print('struct.__dict__.keys()', struct.__dict__.keys())
    if 'Synthetic_Simulations_Archive' not in struct.__dict__.keys(): # create the Simulations Archive for all the Simulations
        print('create the Simulations Archive for all the Simulations')
        # all the attributes of the combination
        val = dict(zip(['all_traces_df', 'mother_df', 'daughter_df', 'm_d_corr_df', 'column_names', 'columns_to_show', 'columns_to_show_m',
                        'columns_to_show_d', 'positive_beta', 'growth_average'],
                       [all_traces_df, mother_df, daughter_df, m_d_corr_df, column_names, columns_to_show, columns_to_show_m, columns_to_show_d,
                        positive_beta, growth_average]))
        updated_dic = dict({key: val})
        struct.Synthetic_Simulations_Archive = updated_dic
        print('put into the synth sim arch')
    elif key not in struct.Synthetic_Simulations_Archive.keys():
        print("we didn't run: "+key)
        # all the attributes of the combination
        val = dict(zip(['all_traces_df', 'mother_df', 'daughter_df', 'm_d_corr_df', 'column_names', 'columns_to_show', 'columns_to_show_m',
                         'columns_to_show_d', 'positive_beta', 'growth_average'],
                        [all_traces_df, mother_df, daughter_df, m_d_corr_df, column_names, columns_to_show, columns_to_show_m, columns_to_show_d,
                         positive_beta, growth_average]))

        print('1', struct.Synthetic_Simulations_Archive.keys())
        print(type(Synthetic_Simulations_Archive), type(struct.Synthetic_Simulations_Archive))
        print(type(key), type(val))

        other_dict = dict({key : val})

        print(other_dict)

        struct.Synthetic_Simulations_Archive.update(other_dict)

        print('2', Synthetic_Simulations_Archive.keys())

    print(struct.Synthetic_Simulations_Archive.keys())

    pickle_out = open(pickle_name+".pickle", "wb")
    pickle.dump(struct, pickle_out)
    pickle_out.close()

    output_heatmap(df=m_d_corr_df[columns_to_show].loc[columns_to_show], title='Mother/Daughter Correlations', x_labels=columns_to_show_m,
                   y_labels=columns_to_show_d)

    """

    ''' choose trace specific betas, time and growth rate averages '''
    # 1: A, 2: B, 3: trap and 4: population
    A_simulated_traces, B_simulated_traces, A_mother_simulated_traces, B_mother_simulated_traces, A_daughter_simulated_traces, \
    B_daughter_simulated_traces = Simulation(dset='Sisters', beta_A_number=0,
               beta_B_number=1, time_A_number=0, time_B_number=1, growth_A_number=0, growth_B_number=1, intercept_A_number=0, intercept_B_number=1)

    # Mother/Daughter, Sister dataset
    m_d_corr_df = correlation_dataframe(A_mother_simulated_traces.append(B_mother_simulated_traces),
                          A_daughter_simulated_traces.append(B_daughter_simulated_traces))

    output_heatmap(df=m_d_corr_df, title='Mother/Daughter Correlations', x_labels=m_d_corr_df.columns, y_labels=m_d_corr_df.columns)

    exit()

    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # simple renaming
    factors = struct.factors
    targets = struct.targets
    X_test = struct.X_test
    X_train = struct.X_train
    y_test = struct.y_test
    y_train = struct.y_train
    m_d_dependance_units = struct.m_d_dependance_units
    cols = struct.cols
    pop_means = struct.pop_means
    pop_stds = struct.pop_stds

    # # to show the different betas
    # graph_different_betas(struct, pop_means, do_I_print = False, do_I_pool_both_traces = True)
    # exit()

    """


def normalize_the_covariance(matrix):

    # check if it is square and dimension bigger than just 1
    if len(matrix.shape) < 2 or matrix.shape[0] != matrix.shape[1]:
        IOError('Either not square or the dimension is smaller than 2')

    new_mat = np.ones_like(matrix)
    # normalize the matrix
    for row_num in np.arange(matrix.shape[0]):
        for col_num in np.arange(matrix.shape[1]):
            new_mat[row_num, col_num] = matrix[row_num, col_num] / np.sqrt(np.abs(matrix[row_num, row_num])*np.abs(matrix[col_num, col_num]))

    return new_mat


''' alpha, beta, a, b, P^2 and S^2 as specified in Lukas's formalism '''
def theoretic_correlations(alpha, beta, a, b, P, S):

    A_matrix = np.array([[-beta, -beta / alpha], [alpha, 1]])

    beta_coefficient = 1 / (-(beta ** 2) + 2 * beta)

    stationary_covariance = np.array([[(P ** 2) * (2 * beta) + (S ** 2) * ((beta ** 2) / (alpha ** 2)),
                                       (P ** 2) * (-beta * alpha) + (S ** 2) * (-beta / alpha)],
                                      [(P ** 2) * (-beta * alpha) + (S ** 2) * (-beta / alpha),
                                       (P ** 2) * (alpha ** 2) + (S ** 2) * (-(beta ** 2) + 2 * beta + 1)]])

    stationary_covariance = stationary_covariance * beta_coefficient

    transpose_power_b = np.linalg.matrix_power(A_matrix.T, b)

    the_two_matrices_on_the_right = np.matmul(stationary_covariance, transpose_power_b)

    A_power_a = np.linalg.matrix_power(A_matrix, a)

    correlation_matrix = np.matmul(A_power_a, the_two_matrices_on_the_right)

    covariance_matrix = np.ones_like(stationary_covariance)

    print('variance of delta phi:', stationary_covariance[0][0])
    print('variance of y:', stationary_covariance[1][1])

    if a == 0 and b == 0: # same cell
        covariance_matrix = normalize_the_covariance(stationary_covariance)
    else: # mother/daughter
        for row in range(stationary_covariance.shape[0]):
            for col in range(stationary_covariance.shape[1]):
                covariance_matrix[row, col] = (correlation_matrix[row, col] / np.sqrt(np.abs(stationary_covariance[row, row]) *
                                                                                      np.abs(stationary_covariance[col, col])))

    return covariance_matrix


''' 
Here we do the simulation of Naama and Lukas's model and compare it to what we get from the data 
'''
def main():
    # # if you wanna run the simulation
    # run_the_simulation(pickle_name='metastructdata', number_of_traces = 100, cycles_per_trace = 100, c_1 = 50, c_2 = 50, growth_average = 1.32,
    #                    positive_beta = None)
    # run_the_simulation(pickle_name='metastructdata_old')

    pickle_name = 'metastructdata'
    # Import the Refined Data
    pickle_in = open(pickle_name + ".pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    m_d_dependance_units = struct.m_d_dependance_units

    '''
    the keys are:
    ['all_traces_df', 'mother_df', 'daughter_df', 'm_d_corr_df', 'column_names', 'columns_to_show',
    'columns_to_show_m', 'columns_to_show_d', 'positive_beta', 'growth_average']
    '''
    collection = struct.Synthetic_Simulations_Archive['100, 100, 50, 50']

    P=.2
    S=.1

    all_traces_df, mother_df, daughter_df = completely_synthetic_simulation(growth_average=1, beta=.485, c_1=.2, c_2=.1, number_of_traces=1,
                                                                          cycles_per_trace=1000, column_names=
    ['generationtime', 'log-normalized_length_birth', 'growth_length', 'division_ratios__f_n', 'log-normalized_length_final', 'phi'])

    print('simulation of log-norm length \n', np.cov(mother_df['log-normalized_length_birth'], daughter_df['log-normalized_length_birth']))
    print('theoretical expression of log-norm length \n', )
    print('simulation of delta phi \n', np.cov(mother_df['phi'], daughter_df['phi']))
    exit()

    # # replacing phi with generationtime
    # regPopulation = LR().fit(
    #     np.array(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m']))).reshape(-1, 1),
    #     np.array(m_d_dependance_units['generationtime_m'] - np.mean(m_d_dependance_units['generationtime_m'])).reshape(-1, 1))
    # betaPopulation = regPopulation.coef_[0][0]
    # interceptPopulation = regPopulation.intercept_

    regPopulation = LR().fit(
        np.array(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m']))).reshape(-1, 1),
        np.array(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m'])).reshape(-1, 1))
    betaPopulation = regPopulation.coef_[0][0]
    interceptPopulation = regPopulation.intercept_

    # # Plotting the population phi vs log normalized length and the best fit, beta
    # LBs = np.array(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])))
    # plt.scatter(LBs,
    #             np.array(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m'])), label=stats.pearsonr(
    #         LBs,
    #         np.array(m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m']))
    #     ))
    # plt.legend()
    # plt.plot(np.linspace(min(LBs), max(LBs)), interceptPopulation + betaPopulation * np.linspace(min(LBs), max(LBs)), color='orange')
    # plt.show()

    # print(-betaPopulation)

    mother_data = np.array([m_d_dependance_units['phi_m'] - np.mean(m_d_dependance_units['phi_m']),
                       np.array(np.log(m_d_dependance_units['length_birth_m'] / np.mean(m_d_dependance_units['length_birth_m'])))])

    # # IF we are a saved simulation
    # mother_model = np.array([np.vstack(collection['mother_df']['phi']).flatten(), np.vstack(collection['mother_df'][
    #                                                                                             'log-normalized_length_birth']).flatten()])

    mother_model = np.array([np.vstack(mother_df['phi']).flatten(), np.vstack(mother_df['log-normalized_length_birth']).flatten()])

    print('Data Same-Cell covariance Matrix \n', np.corrcoef(mother_data, mother_data))

    print('Model Same-Cell covariance Matrix \n', np.corrcoef(mother_model, mother_model))
    print('BLAHHH Model Same-Cell covariance Matrix \n', np.cov(mother_model, mother_model))

    covariance_matrix = theoretic_correlations(alpha=1, beta=.485, a=0, b=0, P=.1, S=.2)

    print('Theory Same-Cell covariance Matrix \n', covariance_matrix)

    # exit()

    # print('Theory Same-Cell covariance Matrix \n', pd.DataFrame(covariance_matrix, columns=['Delta Fold Growth', 'Normalized Birth Length']))

    daughter_data = np.array([m_d_dependance_units['phi_d'],
                             np.array(np.log(m_d_dependance_units['length_birth_d'] / np.mean(m_d_dependance_units['length_birth_d'])))])

    # daughter_model = np.array([np.vstack(collection['daughter_df']['phi']).flatten(), np.vstack(collection['daughter_df'][
    #                                                                                             'log-normalized_length_birth']).flatten()])

    daughter_model = np.array([np.vstack(daughter_df['phi']).flatten(), np.vstack(daughter_df['log-normalized_length_birth']).flatten()])

    print('Data Mother/Daughter covariance Matrix \n', np.corrcoef(mother_data, daughter_data))

    print('Model Mother/Daughter covariance Matrix \n', np.corrcoef(mother_model, daughter_model))

    covariance_matrix = theoretic_correlations(alpha=1.32, beta=-betaPopulation, a=1, b=0, P=.1, S=.1)

    print('Theory Mother/Daughter covariance Matrix \n', covariance_matrix)

    # print('Model Mother/Daughter covariance Matrix \n', pd.DataFrame(covariance_matrix, columns=['Delta Fold Growth', 'Normalized Birth Length']))

    exit()
    symbol_labels_m = [r'$\tau$', r'$ln(\frac{x(0)}{x^*})$', r'$ln(\frac{x(\tau)}{x^*})$']
    symbol_labels_d = [r'$\tau$', r'$ln(\frac{x(0)}{x^*})$', r'$ln(\frac{x(\tau)}{x^*})$']

    df = correlation_dataframe(collection['mother_df'][['generationtime', 'log-normalized_length_birth', 'log-normalized_length_final']],
                               collection['mother_df'][['generationtime', 'log-normalized_length_birth', 'log-normalized_length_final']])  # can be
    # columns_d

    output_heatmap(df, title='Same-Cell Correlations (Model)', x_labels=symbol_labels_m, y_labels=symbol_labels_d)


    exit()



    symbol_labels_m = [r'$\tau_{mom}$', r'$ln(\frac{x(0)}{x^*})_{mom}$', r'$ln(\frac{x(\tau)}{x^*})_{mom}$']
    symbol_labels_d = [r'$\tau_{daughter}$', r'$ln(\frac{x(0)}{x^*})_{daughter}$', r'$ln(\frac{x(\tau)}{x^*})_{daughter}$']

    df = correlation_dataframe(collection['mother_df'][['generationtime', 'log-normalized_length_birth', 'log-normalized_length_final']],
                               collection['daughter_df'][['generationtime', 'log-normalized_length_birth', 'log-normalized_length_final']])  # can be
    # columns_d

    output_heatmap(df, title='Mother/Daughter Correlations (Model)', x_labels=symbol_labels_m, y_labels=symbol_labels_d)


if __name__ == '__main__':
    main() # for both run it
