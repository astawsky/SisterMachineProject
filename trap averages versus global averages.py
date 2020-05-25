import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from correlation_dataframe_and_heatmap import output_heatmap
from correlation_dataframe_and_heatmap import correlation_dataframe
import pickle
import scipy.stats as stats

'''
c_1 and c_2 are the coefficients of noise terms
the number of traces we will manufacture
how many cycles we will record per trace
'''


def completely_synthetic_simulation(growth_average, beta, c_1=0.2, c_2=0.2, number_of_traces=800, cycles_per_trace=50, column_names=
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
        if np.mod(mother_index + 1, cycles_per_trace) != 0:
            daughter_df = daughter_df.append(all_traces_df.loc[mother_index + 1])
            mother_df = mother_df.append(all_traces_df.loc[mother_index])

    return all_traces_df, mother_df, daughter_df


def first_try():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    columns_m = ['generationtime_m', 'length_birth_m', 'length_final_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m']
    columns_d = ['generationtime_d', 'length_birth_d', 'length_final_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d']
    trap_avg_columns = ['trap_average_generationtime', 'trap_average_length_birth',
                        'trap_average_length_final', 'trap_average_growth_length', 'trap_average_division_ratios__f_n', 'trap_average_phi']

    # Rename
    m_d_dependance = struct.m_d_dependance_units

    # make the lengths log global-average centered
    m_d_dependance['length_birth_m'] = np.log(m_d_dependance['length_birth_m'] / struct.x_avg)
    m_d_dependance['length_final_m'] = np.log(m_d_dependance['length_final_m'] / struct.x_avg)
    m_d_dependance['length_birth_d'] = np.log(m_d_dependance['length_birth_d'] / struct.x_avg)
    m_d_dependance['length_final_d'] = np.log(m_d_dependance['length_final_d'] / struct.x_avg)

    ''' labels that look nice and describe what we're showing '''
    daughter_symbolic_labels = [r'$\Delta \tau_{n+1, trap}$', r'$\Delta ln(\frac{x_{n+1, trap}(0)}{x^*})$',
                                r'$\Delta ln(\frac{x_{n+1, trap}(\tau)}{x^*})$',
                                r'$\Delta \alpha_{n+1, trap}$', r'$\Delta f_{n+1, trap}$', r'$\Delta \phi_{n+1, trap}$',
                                r'$\Delta \tau_{n+1, global}$', r'$\Delta ln(\frac{x_{n+1, global}(0)}{x^*})$',
                                r'$\Delta ln(\frac{x_{n+1, global}(\tau)}{x^*})$',
                                r'$\Delta \alpha_{n+1, global}$', r'$\Delta f_{n+1, global}$', r'$\Delta \phi_{n+1, global}$']
    mother_symbolic_labels = [r'$\Delta \tau_{n, trap}$', r'$\Delta ln(\frac{x_{n, trap}(0)}{x^*})$',
                              r'$\Delta ln(\frac{x_{n, trap}(\tau)}{x^*})$',
                              r'$\Delta \alpha_{n, trap}$', r'$\Delta f_{n, trap}$', r'$\Delta \phi_{n, trap}$',
                              r'$\Delta \tau_{n, global}$', r'$\Delta ln(\frac{x_{n, global}(0)}{x^*})$',
                              r'$\Delta ln(\frac{x_{n, global}(\tau)}{x^*})$',
                              r'$\Delta \alpha_{n, global}$', r'$\Delta f_{n, global}$', r'$\Delta \phi_{n, global}$']

    same_symbolic_labels = [r'$\Delta \tau_{trap}$', r'$\Delta ln(\frac{x_{trap}(0)}{x^*})$',
                            r'$\Delta ln(\frac{x_{trap}(\tau)}{x^*})$',
                            r'$\Delta \alpha_{trap}$', r'$\Delta f_{trap}$', r'$\Delta \phi_{trap}$',
                            r'$\Delta \tau_{global}$', r'$\Delta ln(\frac{x_{global}(0)}{x^*})$',
                            r'$\Delta ln(\frac{x_{global}(\tau)}{x^*})$',
                            r'$\Delta \alpha_{global}$', r'$\Delta f_{global}$', r'$\Delta \phi_{global}$']

    trap_avg_symbolic_labels = [r'$<\tau_{trap}>$', r'$<ln(\frac{x_{trap}(0)}{x^*})>$',
                                r'$<ln(\frac{x_{trap}(\tau)}{x^*})>$',
                                r'$<\alpha_{trap}>$', r'$<f_{trap}>$', r'$<\phi_{trap}>$']

    # making the trap average dataframes
    trap_avgs_m = m_d_dependance[trap_avg_columns].rename(columns=dict(zip(trap_avg_columns, columns_m)))
    trap_avgs_d = m_d_dependance[trap_avg_columns].rename(columns=dict(zip(trap_avg_columns, columns_d)))
    mother_trap_df = m_d_dependance[columns_m] - trap_avgs_m
    mother_trap_df = mother_trap_df.rename(columns=dict(zip(columns_m, mother_symbolic_labels[:6])))
    daughter_trap_df = m_d_dependance[columns_d] - trap_avgs_d
    daughter_trap_df = daughter_trap_df.rename(columns=dict(zip(columns_d, daughter_symbolic_labels[:6])))

    # making the global average dataframes
    mother_global_df = m_d_dependance[columns_m] - np.mean(m_d_dependance[columns_m])
    mother_global_df = mother_global_df.rename(columns=dict(zip(columns_m, mother_symbolic_labels[6:])))
    daughter_global_df = m_d_dependance[columns_d] - np.mean(m_d_dependance[columns_d])
    daughter_global_df = daughter_global_df.rename(columns=dict(zip(columns_d, daughter_symbolic_labels[6:])))

    # concatenating them to get the 2x as big matrix
    mother_df = pd.concat([mother_trap_df, mother_global_df], axis=1)
    daughter_df = pd.concat([daughter_trap_df, daughter_global_df], axis=1)
    labels_same = same_symbolic_labels
    labels_m = mother_symbolic_labels
    labels_d = daughter_symbolic_labels

    # NEW IDEA correlations between the variable trap averages
    print(trap_avgs_m)
    # exit()

    for col, title in zip(trap_avgs_m.columns, trap_avg_symbolic_labels):
        plt.hist(trap_avgs_m[col], weights=np.ones_like(trap_avgs_m[col]) / float(len(trap_avgs_m[col])), range=
            [np.mean(trap_avgs_m[col])-2*np.std(trap_avgs_m[col]), np.mean(trap_avgs_m[col])+2*np.std(trap_avgs_m[col])])
        plt.title(title)
        # plt.savefig(str(col)+' trap average distribution', dpi=300)
        plt.show()
        plt.close()

    df = correlation_dataframe(trap_avgs_m, trap_avgs_m)

    print(df)

    output_heatmap(df, title='Trap-Average Correlations', x_labels=trap_avg_symbolic_labels, y_labels=trap_avg_symbolic_labels)

    # same-cell correlations

    df = correlation_dataframe(mother_df, mother_df)  # could also be daughter_df

    print(df)

    output_heatmap(df, title='Same-Cell Correlations', x_labels=labels_same, y_labels=labels_same)

    # mother and daughter

    df = correlation_dataframe(mother_df, daughter_df)

    print(df)

    output_heatmap(df, title='mother daughter correlations', x_labels=labels_m, y_labels=labels_d)


def second_try():
    # Import the Refined Data
    pickle_in = open("metastructdata.pickle", "rb")
    struct = pickle.load(pickle_in)
    pickle_in.close()

    # Sister correlation heatmaps from the three datasets

    sis_A = pd.DataFrame(columns=struct.A_dict_sis['Sister_Trace_A_0'].columns)
    sis_B = pd.DataFrame(columns=struct.A_dict_sis['Sister_Trace_A_0'].columns)
    non_sis_A = pd.DataFrame(columns=struct.A_dict_sis['Sister_Trace_A_0'].columns)
    non_sis_B = pd.DataFrame(columns=struct.A_dict_sis['Sister_Trace_A_0'].columns)
    control_A = pd.DataFrame(columns=struct.A_dict_sis['Sister_Trace_A_0'].columns)
    control_B = pd.DataFrame(columns=struct.A_dict_sis['Sister_Trace_A_0'].columns)

    gen_relation = 0

    for col in sis_A.columns:
        sis_A[col] = np.array([trace_A[col][gen_relation] for trace_A, trace_B in zip(struct.A_dict_sis.values(),
                     struct.B_dict_sis.values())])
        sis_B[col] = np.array([trace_B[col][gen_relation] for trace_A, trace_B in zip(struct.A_dict_sis.values(),
                                                                                                                struct.B_dict_sis.values())])
        non_sis_A[col] = np.array([trace_A[col][gen_relation] for trace_A, trace_B in zip(struct.A_dict_non_sis.values(),
                                                                                                            struct.B_dict_non_sis.values())])
        non_sis_B[col] = np.array([trace_B[col][gen_relation] for trace_A, trace_B in zip(struct.A_dict_non_sis.values(),
                                                                                                            struct.B_dict_non_sis.values())])
        control_A[col] = np.array([trace_A[col][gen_relation] for trace_A, trace_B in zip(struct.A_dict_both.values(),
                                                                                                                struct.B_dict_both.values())])
        control_B[col] = np.array([trace_B[col][gen_relation] for trace_A, trace_B in zip(struct.A_dict_both.values(),
                                                                                                                struct.B_dict_both.values())])
    
    # sis_A['length_birth'] = np.log(sis_A['length_birth'])
    # sis_B['length_birth'] = np.log(sis_B['length_birth'])
    # non_sis_A['length_birth'] = np.log(non_sis_A['length_birth'])
    # non_sis_B['length_birth'] = np.log(non_sis_B['length_birth'])
    # control_A['length_birth'] = np.log(control_A['length_birth'])
    # control_B['length_birth'] = np.log(control_B['length_birth'])
    # sis_A['length_final'] = np.log(sis_A['length_final'])
    # sis_B['length_final'] = np.log(sis_B['length_final'])
    # non_sis_A['length_final'] = np.log(non_sis_A['length_final'])
    # non_sis_B['length_final'] = np.log(non_sis_B['length_final'])
    # control_A['length_final'] = np.log(control_A['length_final'])
    # control_B['length_final'] = np.log(control_B['length_final'])

    combined_sis_A_df = sis_A.append(sis_B)
    combined_non_sis_A_df = non_sis_A.append(non_sis_B)
    combined_control_A_df = control_A.append(control_B)

    combined_sis_B_df = sis_B.append(sis_A)
    combined_non_sis_B_df = non_sis_B.append(non_sis_A)
    combined_control_B_df = control_B.append(control_A)

    sis_df = correlation_dataframe(combined_sis_A_df, combined_sis_B_df)
    non_sis_df = correlation_dataframe(combined_non_sis_A_df, combined_non_sis_B_df)
    control_df = correlation_dataframe(combined_control_A_df, combined_control_B_df)
    
    # sis_df = correlation_dataframe(sis_A, sis_B)
    # non_sis_df = correlation_dataframe(non_sis_A, non_sis_B)
    # control_df = correlation_dataframe(control_A, control_B)
    
    lbls_A = [r'$\tau_{A}$', r'$ln(\frac{x_{A}(0)}{x^*})$',
                            r'$ln(\frac{x_{A}(\tau)}{x^*})$',
                            r'$\alpha_{A}$', r'$f_{A}$', r'$\phi_{A}$']
    lbls_B = [r'$\tau_{B}$', r'$ln(\frac{x_{B}(0)}{x^*})$',
              r'$ln(\frac{x_{B}(\tau)}{x^*})$',
              r'$\alpha_{B}$', r'$f_{B}$', r'$\phi_{B}$']

    # # To see why we got this weird asymmetry in the sister correlation heatmap comparison of S and NS
    #
    # memory = []
    # for var1 in sis_df.columns:
    #     for var2 in sis_df.columns:
    #         if [var1, var2] not in memory and [var2, var1] not in memory:
    #             m, c = stats.linregress(combined_sis_A_df[var1], combined_sis_B_df[var2])[:2]
    #             print(m, c)
    #             x_range = [np.mean(combined_sis_A_df[var1])-3*np.std(combined_sis_A_df[var1]),
    #                       np.mean(combined_sis_A_df[var1])+3*np.std(combined_sis_A_df[var1])]
    #             y_range = [np.mean(combined_sis_B_df[var2]) - 3 * np.std(combined_sis_B_df[var2]),
    #                       np.mean(combined_sis_B_df[var2]) + 3 * np.std(combined_sis_B_df[var2])]
    #             plt.scatter(combined_sis_A_df[var1], combined_sis_B_df[var2], label=str(round(stats.pearsonr(combined_sis_A_df[var1],
    #                                                                                                         combined_sis_B_df[var2])[0], 3)))
    #             plt.plot(c+m*np.linspace(np.mean(combined_sis_A_df[var1])-3*np.std(combined_sis_A_df[var1]),
    #                       np.mean(combined_sis_A_df[var1])+3*np.std(combined_sis_A_df[var1])), color='orange')
    #             plt.xlim(x_range)
    #             plt.ylim(y_range)
    #             plt.xlabel(str(var1))
    #             plt.ylabel(str(var2))
    #             plt.legend()
    #             plt.show()
    #
    # exit()
    #
    # plt.scatter(sis_A['length_birth'], sis_B['length_final'], label=stats.pearsonr(sis_A['length_birth'], sis_B['length_final'])[0])
    # plt.legend()
    # plt.show()
    #
    # plt.scatter(non_sis_A['length_birth'], non_sis_B['length_final'], label=stats.pearsonr(non_sis_A['length_birth'], non_sis_B['length_final'])[0])
    # plt.legend()
    # plt.show()
    #
    # plt.scatter(sis_A['length_final'], sis_B['length_birth'], label=stats.pearsonr(sis_A['length_final'], sis_B['length_birth'])[0])
    # plt.legend()
    # plt.show()
    #
    # plt.scatter(non_sis_A['length_final'], non_sis_B['length_birth'], label=stats.pearsonr(non_sis_A['length_final'], non_sis_B['length_birth'])[0])
    # plt.legend()
    # plt.show()

    print(sis_A.columns)
    print(lbls_A)

    output_heatmap(sis_df, title='Sister Correlations (S dataset)', x_labels=lbls_A, y_labels=lbls_B)
    output_heatmap(non_sis_df, title='Sister Correlations (NS dataset)', x_labels=lbls_A, y_labels=lbls_B)
    output_heatmap(control_df, title='Sister Correlations (C dataset)', x_labels=lbls_A, y_labels=lbls_B)


    columns_m = ['generationtime_m', 'length_birth_m', 'length_final_m', 'growth_length_m', 'division_ratios__f_n_m', 'phi_m']
    columns_d = ['generationtime_d', 'length_birth_d', 'length_final_d', 'growth_length_d', 'division_ratios__f_n_d', 'phi_d']
    trap_avg_columns = ['trap_average_generationtime', 'trap_average_length_birth',
                        'trap_average_length_final', 'trap_average_growth_length', 'trap_average_division_ratios__f_n', 'trap_average_phi']

    # Rename
    m_d_dependance = struct.m_d_dependance_units

    # make the lengths log
    m_d_dependance['length_birth_m'] = np.log(m_d_dependance['length_birth_m'])
    m_d_dependance['length_final_m'] = np.log(m_d_dependance['length_final_m'])
    m_d_dependance['length_birth_d'] = np.log(m_d_dependance['length_birth_d'])
    m_d_dependance['length_final_d'] = np.log(m_d_dependance['length_final_d'])

    ''' labels that look nice and describe what we're showing '''
    daughter_symbolic_labels = [r'$\Delta \tau_{n+1, trap}$', r'$\Delta ln(\frac{x_{n+1, trap}(0)}{x^*})$',
                                r'$\Delta ln(\frac{x_{n+1, trap}(\tau)}{x^*})$',
                                r'$\Delta \alpha_{n+1, trap}$', r'$\Delta f_{n+1, trap}$', r'$\Delta \phi_{n+1, trap}$',
                                r'$\Delta \tau_{n+1, global}$', r'$\Delta ln(\frac{x_{n+1, global}(0)}{x^*})$',
                                r'$\Delta ln(\frac{x_{n+1, global}(\tau)}{x^*})$',
                                r'$\Delta \alpha_{n+1, global}$', r'$\Delta f_{n+1, global}$', r'$\Delta \phi_{n+1, global}$']
    mother_symbolic_labels = [r'$\Delta \tau_{n, trap}$', r'$\Delta ln(\frac{x_{n, trap}(0)}{x^*})$',
                              r'$\Delta ln(\frac{x_{n, trap}(\tau)}{x^*})$',
                              r'$\Delta \alpha_{n, trap}$', r'$\Delta f_{n, trap}$', r'$\Delta \phi_{n, trap}$',
                              r'$\Delta \tau_{n, global}$', r'$\Delta ln(\frac{x_{n, global}(0)}{x^*})$',
                              r'$\Delta ln(\frac{x_{n, global}(\tau)}{x^*})$',
                              r'$\Delta \alpha_{n, global}$', r'$\Delta f_{n, global}$', r'$\Delta \phi_{n, global}$']

    same_symbolic_labels = [r'$\Delta \tau_{trap}$', r'$\Delta ln(\frac{x_{trap}(0)}{x^*})$',
                            r'$\Delta ln(\frac{x_{trap}(\tau)}{x^*})$',
                            r'$\Delta \alpha_{trap}$', r'$\Delta f_{trap}$', r'$\Delta \phi_{trap}$',
                            r'$\Delta \tau_{global}$', r'$\Delta ln(\frac{x_{global}(0)}{x^*})$',
                            r'$\Delta ln(\frac{x_{global}(\tau)}{x^*})$',
                            r'$\Delta \alpha_{global}$', r'$\Delta f_{global}$', r'$\Delta \phi_{global}$']

    trap_avg_symbolic_labels = [r'$<\tau_{trap}>$', r'$<ln(\frac{x_{trap}(0)}{x^*})>$',
                                r'$<ln(\frac{x_{trap}(\tau)}{x^*})>$',
                                r'$<\alpha_{trap}>$', r'$<f_{trap}>$', r'$<\phi_{trap}>$']

    # making the trap average dataframes
    trap_avgs_m = m_d_dependance[trap_avg_columns].rename(columns=dict(zip(trap_avg_columns, columns_m)))
    trap_avgs_d = m_d_dependance[trap_avg_columns].rename(columns=dict(zip(trap_avg_columns, columns_d)))
    # make the lengths log
    trap_avgs_m['length_birth_m'] = np.log(trap_avgs_m['length_birth_m'])
    trap_avgs_m['length_final_m'] = np.log(trap_avgs_m['length_final_m'])
    trap_avgs_d['length_birth_d'] = np.log(trap_avgs_d['length_birth_d'])
    trap_avgs_d['length_final_d'] = np.log(trap_avgs_d['length_final_d'])
    # minus the trap averages
    mother_trap_df = m_d_dependance[columns_m] - trap_avgs_m
    mother_trap_df = mother_trap_df.rename(columns=dict(zip(columns_m, mother_symbolic_labels[:6])))
    daughter_trap_df = m_d_dependance[columns_d] - trap_avgs_d
    daughter_trap_df = daughter_trap_df.rename(columns=dict(zip(columns_d, daughter_symbolic_labels[:6])))

    # making the global average dataframes
    mother_global_df = m_d_dependance[columns_m] - np.mean(m_d_dependance[columns_m])
    mother_global_df = mother_global_df.rename(columns=dict(zip(columns_m, mother_symbolic_labels[6:])))
    daughter_global_df = m_d_dependance[columns_d] - np.mean(m_d_dependance[columns_d])
    daughter_global_df = daughter_global_df.rename(columns=dict(zip(columns_d, daughter_symbolic_labels[6:])))

    # concatenating them to get the 2x as big matrix
    mother_df = pd.concat([mother_trap_df, mother_global_df], axis=1)
    daughter_df = pd.concat([daughter_trap_df, daughter_global_df], axis=1)
    labels_same = same_symbolic_labels
    labels_m = mother_symbolic_labels
    labels_d = daughter_symbolic_labels

    # NEW IDEA correlations between the variable trap averages

    for col, title in zip(trap_avgs_m.columns, trap_avg_symbolic_labels):
        plt.hist(trap_avgs_m[col], weights=np.ones_like(trap_avgs_m[col]) / float(len(trap_avgs_m[col])), range=
        [np.mean(trap_avgs_m[col]) - 2 * np.std(trap_avgs_m[col]), np.mean(trap_avgs_m[col]) + 2 * np.std(trap_avgs_m[col])])
        plt.title(title)
        # plt.savefig('second try '+str(col)+' trap average distribution', dpi=300)
        plt.show()
        plt.close()

    df = correlation_dataframe(trap_avgs_m, trap_avgs_m)

    print(df)

    output_heatmap(df, title='Trap-Average Correlations', x_labels=trap_avg_symbolic_labels, y_labels=trap_avg_symbolic_labels)

    # same-cell correlations

    df = correlation_dataframe(mother_df, mother_df)  # could also be daughter_df

    print(df)

    output_heatmap(df, title='Same-Cell Correlations', x_labels=labels_same, y_labels=labels_same)

    # mother and daughter

    df = correlation_dataframe(mother_df, daughter_df)

    print(df)

    output_heatmap(df, title='mother daughter correlations', x_labels=labels_m, y_labels=labels_d)


def main():
    # first_try()
    second_try()


if __name__ == '__main__':
    main()
