import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle
import os
import scipy.stats as stats
import random
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import QuantileTransformer, Binarizer, KBinsDiscretizer
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, cluster
import seaborn as sns
from SisterMachineDataPipeline import SisterCellData
import iteround
import matplotlib as mpl


def print_full_dataframe(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)


def create_folder(filename):
    # create the folder if not created already
    try:
        # Create target Directory
        os.mkdir(filename)
        print("Directory", filename, "Created ")
    except FileExistsError:
        print("Directory", filename, "already exists")


def original_bootstraped_MI_and_jointprobs_and_ci(label_A_vector, label_B_vector, how_many_times_to_resample, **kwargs):
    equalize_number = kwargs.get('equalize_number', len(label_A_vector))  # what the size of the vectors will be
    c_i = kwargs.get('confidence_interval', None)  # No confidence interval will be given if None
    calculate_acceleration = kwargs.get('calculate_acceleration', False)
    show_histograms = kwargs.get('show_histograms', False)
    percent_to_cover = kwargs.get('percent_to_cover', .9)
    show_joint_probability = kwargs.get('show_joint_probability', False)
    return_array = []

    # warnings
    if (percent_to_cover >= 1) or (percent_to_cover <= 0):
        raise IOError('percent_to_cover must be strictly between 0 and 1')

    # trim the vector to the equalize_number and reset the index
    label_A_vector = label_A_vector.sample(n=equalize_number, replace=False, axis='index')
    label_B_vector = label_B_vector.loc[label_A_vector.index].reset_index(drop=True)
    label_A_vector = label_A_vector.reset_index(drop=True)

    # get the MIs with the original dataset
    original_MI = {'MI': mutual_info_score(label_A_vector, label_B_vector), 'Normalized MI': normalized_mutual_info_score(label_A_vector, label_B_vector, average_method='arithmetic'),
                   'Adjusted MI': adjusted_mutual_info_score(label_A_vector, label_B_vector, average_method='arithmetic')}

    # returning the original MIs that always comes first
    return_array.append(original_MI)

    # !!!bootstrap and calculate the Mutual Information on this resampled dataset with replacement!!!
    bootstrap_dictionary = {'MI': [], 'Normalized MI': [], 'Adjusted MI': []}
    percentage_of_original_indices = []
    joint_probabilities_array = []
    for resample_number in range(how_many_times_to_resample):
        # resampled with replacement data
        A_new = label_A_vector.sample(n=len(label_A_vector), replace=True, axis='index')
        B_new = label_B_vector.loc[A_new.index]

        # calculate the joint probabilities matrix and convert it to dictionary to make it simpler
        joint_probabilities_array.append(joint_probability_matrix(np.array(A_new), np.array(B_new), **kwargs).to_dict())

        percentage_of_original_indices.append(len(np.unique(A_new.index)) / len(label_A_vector))

        bootstrap_dictionary['MI'].append(mutual_info_score(A_new, B_new))
        bootstrap_dictionary['Normalized MI'].append(normalized_mutual_info_score(A_new, B_new, average_method='arithmetic'))
        bootstrap_dictionary['Adjusted MI'].append(adjusted_mutual_info_score(A_new, B_new, average_method='arithmetic'))
    # add the original MI to the bootsrap array
    for key in bootstrap_dictionary.keys():
        bootstrap_dictionary[key].append(original_MI[key])

    # returning the bootstrap dictionary that always comes second
    return_array.append(bootstrap_dictionary)

    # get the mean joint probability matrix
    columns_all = []
    rows_all = []
    for ind in range(len(joint_probabilities_array)):
        for column in joint_probabilities_array[ind].keys():
            columns_all.append(column)
            for row in joint_probabilities_array[ind][column].keys():
                rows_all.append(row)

    rows = np.unique(rows_all)
    columns = np.unique(columns_all)
    mean_joint_probabilities = pd.DataFrame.from_dict({column: {row: np.array([joint_probabilities_array[ind][column][row] for ind in range(len(joint_probabilities_array)) if
                                                                               (column in list(joint_probabilities_array[ind].keys())) and (
                                                                                           row in list(joint_probabilities_array[ind][column].keys()))]).sum() / (
                                                                                 how_many_times_to_resample * len(label_A_vector)) for row in rows} for column in columns})

    # To show/save the joint probability matrix
    if show_joint_probability:
        dataset = kwargs.get('dataset', 'Not-Specified')
        directory = kwargs.get('directory', None)

        # automating this so that the heatmap will not look too cramped with the numbers inside the cells
        if np.array(mean_joint_probabilities).shape[0]*np.array(mean_joint_probabilities).shape[1] > 100:
            annot = False
        else:
            annot = True
        # plt.figure(figsize=(12.7, 7.5)) # maybe take this out?
        sns.heatmap(saferound_the_df(mean_joint_probabilities, precision=2), annot=annot, fmt='.2f')
        plt.ylabel('B trajectory bins')
        plt.xlabel('A trajectory bins')
        plt.title("{} Joint probability matrix form {} bootstraps, adds up to {}".format(dataset, how_many_times_to_resample, round(np.array(mean_joint_probabilities).sum()), 3))
        plt.tight_layout()
        # plt.savefig(directory+"/{} Joint probability matrix form {} bootstraps, adds up to {}".format(dataset, how_many_times_to_resample, round(np.array(mean_joint_probabilities).sum()), 3), dpi=300)
        plt.show()
        plt.close()

    # return the mean joint_probabilities matrix, that always comes third
    return_array.append(mean_joint_probabilities)

    # for the percentile points
    alpha = (1.0 - percent_to_cover) / 2.0
    
    if c_i == 'bca':
        # decide if we want to calculate the acceleration or if it is already calculated
        if calculate_acceleration:
            print('calculating acceleration')
            # calculate the acceleration, we don't need the bootstrap
            dropped_index_MIs = {'MI': np.array([mutual_info_score(label_A_vector.drop(index=index), label_B_vector.drop(index=index)) for index in np.arange(len(label_A_vector))]),
                                 'Normalized MI': np.array(
                                     [normalized_mutual_info_score(label_A_vector.drop(index=index), label_B_vector.drop(index=index), average_method='arithmetic') for index in
                                      np.arange(len(label_A_vector))]), 'Adjusted MI': np.array(
                    [adjusted_mutual_info_score(label_A_vector.drop(index=index), label_B_vector.drop(index=index), average_method='arithmetic') for index in np.arange(len(label_A_vector))])}
            constant = {key: val.sum() for key, val in dropped_index_MIs.items()}
            a_numerator = {key: np.array([(constant[key] - dropped_MI) ** 3 for dropped_MI in val]).sum() for key, val in dropped_index_MIs.items()}
            a_denominator = {key: 6 * (np.array([(constant[key] - dropped_MI) ** 2 for dropped_MI in val]).sum() ** (1.5)) for key, val in dropped_index_MIs.items()}
            acceleration = {key: a_numerator[key] / a_denominator[key] for key in a_numerator.keys() if a_numerator.keys() == a_denominator.keys()}
            print('-------')
            print('len(dropped_index_MIs):', len(dropped_index_MIs))
            print('constant:', constant)
            print('a_numerator:', a_numerator)
            print('a_denominator:', a_denominator)
            print('a:', acceleration)
            exit()
        else:
            acceleration = {key: 0.00284123 for key in original_MI.keys()}  # gotten from previously calculating them all

        # caculate the confidence intervals based on the bias and acceleration
        bias_z_naught = {key: stats.norm.ppf(len([True for MI in val if MI <= original_MI[key]]) / how_many_times_to_resample) for key, val in bootstrap_dictionary.items()}
        alpha_lower = {key: stats.norm.cdf(val + (val + stats.norm.ppf(alpha)) / (1 - acceleration[key] * (val + stats.norm.ppf(alpha)))) for key, val in bias_z_naught.items()}
        alpha_upper = {key: stats.norm.cdf(val + (val + stats.norm.ppf(1 - alpha)) / (1 - acceleration[key] * (val + stats.norm.ppf(1 - alpha)))) for key, val in bias_z_naught.items()}
        lower_bound = {key: np.percentile(np.sort(val), 100 * alpha_lower[key]) for key, val in bootstrap_dictionary.items()}
        upper_bound = {key: np.percentile(np.sort(val), 100 * alpha_upper[key]) for key, val in bootstrap_dictionary.items()}

        # update the confidence interval dictionary
        return_array.append({key: [lower_bound[key], upper_bound[key]] for key in lower_bound.keys()})

    if c_i == 'percentile':
        # show the different types of confidence intervals to compare
        return_array.append({key: [np.percentile(val, 100 * alpha), np.percentile(val, 100 * (1.0 - alpha))] for key, val in bootstrap_dictionary.items()})

    if show_histograms == True:
        dataset = kwargs.get('dataset', 'Not-Specified')
        directory = kwargs.get('directory', None)
        
        plt.figure(figsize=(12.7, 7.5))
        sns.distplot(percentage_of_original_indices)
        plt.title('Histogram of percentage of unique indices in the resampling with replacements')
        # plt.show()
        plt.savefig(directory + '/Histogram of percentage of unique indices in the resampling with replacements for {} dataset'.format(dataset), dpi=300)
        plt.close()

        for key, val in bootstrap_dictionary.items():
            plt.figure(figsize=(12.7, 7.5))
            sns.regplot(percentage_of_original_indices, val[:-1], label=round(stats.pearsonr(percentage_of_original_indices, val[:-1])[0], 2))
            plt.title('Correlation between percentage of unique indices and Mutual Information')
            plt.xlabel('percentage of unique indices')
            plt.legend()
            plt.ylabel(key)
            # plt.show()
            plt.savefig(directory + '/Correlation between percentage of unique indices and {} for {} dataset'.format(key, dataset), dpi=300)
            plt.close()

            # print(key, np.array(val), type(val[0]))
            plt.figure(figsize=(12.7, 7.5))
            sns.distplot(val)
            # sns.set_style('whitegrid')
            # plt.hist(val, density=True, cumulative=True, bins=int(np.ceil(np.sqrt(len(val)))))
            sorted = np.sort(val)
            left_bndry = np.percentile(val, 100 * alpha)
            right_bndry = np.percentile(val, 100 * (1.0 - alpha))
            plt.title('Histogram of bootstrapped ' + key + ' values of Generationtime')
            # print('---------')
            # print(left_bndry, right_bndry, np.median(sorted))
            # print(BC_a_confidence_interval[key][0], BC_a_confidence_interval[key][1], np.median(sorted))
            # print('Percentile {}% of the sample inside'.format(round(len([num for num in val if (num <= right_bndry) and (num >= left_bndry)]) / how_many_times_to_resample, 2)))
            # print('Bias corrected, {}% of the sample inside'.format(
            #     round(len([num for num in val if (num <= BC_a_confidence_interval[key][1]) and (num >= BC_a_confidence_interval[key][0])]) / how_many_times_to_resample, 2)))
            plt.axvline(left_bndry, color='black', label='{} to {} percentiles'.format(round(alpha, 2), round(1 - alpha, 2)))
            plt.axvline(right_bndry, color='black')
            # plt.axvline(BC_a_confidence_interval[key][0], color='green')
            # plt.axvline(BC_a_confidence_interval[key][1], color='green')
            plt.axvline(np.median(val), color='orange', ls='--', label='Median')
            plt.axvline(original_MI[key], color='red', ls=':', label='Original')
            plt.legend()
            # plt.show()
            plt.savefig(directory + '/{} for {} dataset'.format(key, dataset), dpi=300)
            plt.close()

    return return_array


def script_to_get_Intra_and_Pooled_boxplot(measurements, how_many_times_to_resample, save_bootstrap_df, **kwargs):

    label_them = kwargs.get('label_them', False)
    var = kwargs.get('var', 'generationtime')

    bootstrap_df = pd.DataFrame(columns=['MI', 'Normalized MI', 'Adjusted MI', 'dataset', 'label'])
    joint_probs_dictionary = {'Sisters': {}, 'Non-Sisters': {}, 'Control': {}}

    sis_A_array = measurements.sis_A_intra_gen_bacteria.copy()
    sis_A_array.append(measurements.sis_A_pooled)
    sis_B_array = measurements.sis_B_intra_gen_bacteria.copy()
    sis_B_array.append(measurements.sis_B_pooled)
    non_sis_A_array = measurements.non_sis_A_intra_gen_bacteria.copy()
    non_sis_A_array.append(measurements.non_sis_A_pooled)
    non_sis_B_array = measurements.non_sis_B_intra_gen_bacteria.copy()
    non_sis_B_array.append(measurements.non_sis_B_pooled)
    con_A_array = measurements.con_A_intra_gen_bacteria.copy()
    con_A_array.append(measurements.con_A_pooled)
    con_B_array = measurements.con_B_intra_gen_bacteria.copy()
    con_B_array.append(measurements.con_B_pooled)
    labels = ['Intra {}'.format(ind) for ind in range(len(measurements.sis_A_intra_gen_bacteria))]
    labels.append('Pooled Traps')

    for sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, label in zip(sis_A_array, sis_B_array, non_sis_A_array, non_sis_B_array, con_A_array, con_B_array, labels):
        equalize_number = min(len(sis_A), len(sis_B), len(non_sis_A), len(non_sis_B), len(con_A), len(con_B))
        print(label, equalize_number)
        A, B = label_the_data(set_estimator(sis_A[var], sis_B[var], **kwargs), sis_A[var], sis_B[var], **kwargs)
        original_MI, bootstrap_dictionary, joint_probabilities = original_bootstraped_MI_and_jointprobs_and_ci(A, B,
            how_many_times_to_resample, dataset='Sisters', show_joint_probability=False, equalize_number=equalize_number, **kwargs)
        bootstrap_dictionary.update({'dataset': ['Sisters' for ind in range(how_many_times_to_resample + 1)], 'label': [label for ind in range(how_many_times_to_resample + 1)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=bootstrap_dictionary)], axis=0).reset_index(drop=True)
        joint_probs_dictionary['Sisters'].update({label: joint_probabilities})

        A, B = label_the_data(set_estimator(non_sis_A[var], non_sis_B[var], **kwargs), non_sis_A[var], non_sis_B[var], **kwargs)
        original_MI, bootstrap_dictionary, joint_probabilities = original_bootstraped_MI_and_jointprobs_and_ci(A, B,
            how_many_times_to_resample, dataset='Non-Sisters', show_joint_probability=False, **kwargs)
        bootstrap_dictionary.update({'dataset': ['Non-Sisters' for ind in range(how_many_times_to_resample + 1)], 'label': [label for ind in range(how_many_times_to_resample + 1)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=bootstrap_dictionary)], axis=0).reset_index(drop=True)
        joint_probs_dictionary['Non-Sisters'].update({label: joint_probabilities})

        A, B = label_the_data(set_estimator(con_A[var], con_B[var], **kwargs), con_A[var], con_B[var], **kwargs)
        original_MI, bootstrap_dictionary, joint_probabilities = original_bootstraped_MI_and_jointprobs_and_ci(A, B,
            how_many_times_to_resample, dataset='Control', show_joint_probability=False, **kwargs)
        bootstrap_dictionary.update({'dataset': ['Control' for ind in range(how_many_times_to_resample + 1)], 'label': [label for ind in range(how_many_times_to_resample + 1)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=bootstrap_dictionary)], axis=0).reset_index(drop=True)
        joint_probs_dictionary['Control'].update({label: joint_probabilities})

    if save_bootstrap_df:
        # save bootstrap df so we don't have to go through that all again
        pickle_out = open("Intra and Pooled Bootstrap dataframe.pickle", "wb")
        pickle.dump(bootstrap_df, pickle_out)
        pickle_out.close()

    various_datasets_average_joint_probabilities(joint_probs_dictionary, ['Sisters', 'Non-Sisters', 'Control'], labels, **kwargs)

    ax = sns.boxplot(x="label", y="MI", hue="dataset", palette="Set3", data=bootstrap_df)
    plt.show()
    plt.xticks(rotation=45)
    # plt.savefig('Gentime MI Intra and Pooled, normal size', dpi=300)
    plt.close()

    ax = sns.boxplot(x="label", y="Normalized MI", hue="dataset", palette="Set3", data=bootstrap_df)
    plt.show()
    plt.xticks(rotation=45)
    # plt.savefig('Gentime NMI Intra and Pooled, normal size', dpi=300)
    plt.close()

    ax = sns.boxplot(x="label", y="Adjusted MI", hue="dataset", palette="Set3", data=bootstrap_df)
    plt.show()
    plt.xticks(rotation=45)
    # plt.savefig('Gentime AMI Intra and Pooled, normal size', dpi=300)
    plt.close()


def script_to_get_only_Pooled_boxplot(measurements, how_many_times_to_resample, **kwargs):
    label = 'Pooled Traps'
    bootstrap_df = pd.DataFrame(columns=['MI', 'Normalized MI', 'Adjusted MI', 'dataset', 'label'])

    original_MI, bootstrap_dictionary, BC_a_confidence_interval = original_bootstraped_MI_and_jointprobs_and_ci(measurements.sis_A_pooled['generationtime'],
        measurements.sis_B_pooled['generationtime'], how_many_times_to_resample, **kwargs)
    bootstrap_dictionary.update({'dataset': ['Sisters' for ind in range(how_many_times_to_resample + 1)], 'label': [label for ind in range(how_many_times_to_resample + 1)]})
    bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=bootstrap_dictionary)], axis=0).reset_index(drop=True)

    original_MI, bootstrap_dictionary, BC_a_confidence_interval = original_bootstraped_MI_and_jointprobs_and_ci(measurements.non_sis_A_pooled['generationtime'],
        measurements.non_sis_B_pooled['generationtime'], how_many_times_to_resample, **kwargs)
    bootstrap_dictionary.update({'dataset': ['Non-Sisters' for ind in range(how_many_times_to_resample + 1)], 'label': [label for ind in range(how_many_times_to_resample + 1)]})
    bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=bootstrap_dictionary)], axis=0).reset_index(drop=True)

    original_MI, bootstrap_dictionary, BC_a_confidence_interval = original_bootstraped_MI_and_jointprobs_and_ci(measurements.con_A_pooled['generationtime'],
        measurements.con_B_pooled['generationtime'], how_many_times_to_resample, **kwargs)
    bootstrap_dictionary.update({'dataset': ['Control' for ind in range(how_many_times_to_resample + 1)], 'label': [label for ind in range(how_many_times_to_resample + 1)]})
    bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=bootstrap_dictionary)], axis=0).reset_index(drop=True)

    ax = sns.boxplot(x="label", y="MI", hue="dataset", palette="Set3", data=bootstrap_df)
    # plt.show()
    plt.tight_layout()
    plt.savefig('Gentime MI Pooled, normal size', dpi=300)
    plt.close()

    ax = sns.boxplot(x="label", y="Normalized MI", hue="dataset", palette="Set3", data=bootstrap_df)
    # plt.show()
    plt.tight_layout()
    plt.savefig('Gentime NMI Pooled, normal size', dpi=300)
    plt.close()

    ax = sns.boxplot(x="label", y="Adjusted MI", hue="dataset", palette="Set3", data=bootstrap_df)
    # plt.show()
    plt.tight_layout()
    plt.savefig('Gentime AMI Pooled, normal size', dpi=300)
    plt.close()


def script_to_get_Inter_boxplot(measurements, how_many_times_to_resample, save_bootstrap_df, **kwargs):
    bootstrap_df = pd.DataFrame(columns=['MI', 'Normalized MI', 'Adjusted MI', 'label'])
    mother_array = measurements.mother_dfs.copy()
    daughter_array = measurements.daughter_dfs.copy()
    labels = ['Inter {}'.format(ind) for ind in range(len(measurements.mother_dfs))]

    for mom, daughter, label in zip(mother_array, daughter_array, labels):
        print(label)
        original_MI, bootstrap_dictionary, BC_a_confidence_interval = original_bootstraped_MI_and_jointprobs_and_ci(mom['generationtime'], daughter['generationtime'],
            how_many_times_to_resample, **kwargs)
        bootstrap_dictionary.update({'label': [label for ind in range(how_many_times_to_resample + 1)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=bootstrap_dictionary)], axis=0).reset_index(drop=True)

    if save_bootstrap_df:
        # save bootstrap df so we don't have to go through that all again
        pickle_out = open("Inter Bootstrap dataframe.pickle", "wb")
        pickle.dump(bootstrap_df, pickle_out)
        pickle_out.close()

    ax = sns.boxplot(x="label", y="MI", palette="Set3", data=bootstrap_df)
    plt.show()
    plt.xticks(rotation=45)
    # plt.savefig('Gentime MI Inter, normal size', dpi=300)
    plt.close()

    ax = sns.boxplot(x="label", y="Normalized MI", palette="Set3", data=bootstrap_df)
    plt.show()
    plt.xticks(rotation=45)
    # plt.savefig('Gentime NMI Inter, normal size', dpi=300)
    plt.close()

    ax = sns.boxplot(x="label", y="Adjusted MI", palette="Set3", data=bootstrap_df)
    plt.show()
    plt.xticks(rotation=45)
    # plt.savefig('Gentime AMI Inter, normal size', dpi=300)
    plt.close()


def MI(a, b, **kwargs):
    check = kwargs.get('check', False)  # this is to check our method against sklearn's

    together = pd.DataFrame({'a': a, 'b': b})  # for the joint distribution
    MI = 0
    for a_class in np.unique(a):
        for b_class in np.unique(b):
            a_marginal = len(a[np.where(a == a_class)]) / len(a)
            b_marginal = len(b[np.where(b == b_class)]) / len(b)
            joint = len(together[(together['a'] == a_class) & (together['b'] == b_class)]) / len(together)
            if (joint != 0):
                MI += joint * np.log(joint / (a_marginal * b_marginal))

    if check:
        print("How similar is our MI to sklearn's?\n", MI, mutual_info_score(a, b))

    return MI


def joint_probability_matrix(a, b, **kwargs):
    check = kwargs.get('check', False)  # this is to check our method against sklearn's
    precision = kwargs.get('precision', 3)
    joint_probabilities = kwargs.get('joint_probabilities', 'new')

    if joint_probabilities == 'new':
        joint_probabilities = pd.DataFrame(data=np.zeros((len(np.unique(a)), len(np.unique(b)))), columns=[str(val) for val in np.unique(b)], index=[str(val) for val in np.unique(a)])

    together = pd.DataFrame({'a': a, 'b': b})  # for the joint distribution
    for a_class in np.unique(a):
        for b_class in np.unique(b):
            joint = len(together[(together['a'] == a_class) & (together['b'] == b_class)])
            joint_probabilities[str(b_class)].loc[str(a_class)] = joint_probabilities[str(b_class)].loc[str(a_class)] + joint

    # # Here we round the numbers so that they add up to 1
    # joint_probabilities = pd.DataFrame(data=np.array([iteround.saferound(np.array(joint_probabilities / len(together))[ind, :], precision) for ind in range(np.array(joint_probabilities).shape[0])]),
    #     columns=[str(val) for val in np.unique(b)], index=[str(val) for val in np.unique(a)])

    # # If the joint distribution matrix does not equal exactly 1 there must be a problem!
    # if np.array(joint_probabilities).sum() != 1.0:
    #     IOError("The sum of the joint distribution matrix is {} != 1.0".format(np.array(joint_probabilities).sum()))

    # If the joint distribution matrix does not equal exactly to the amount of samples there must be a problem!
    if np.array(joint_probabilities).sum() != len(together):
        IOError("The sum of the joint distribution matrix is not the number of samples {} != {}".format(np.array(joint_probabilities).sum(), len(together)))

    # Debugging
    if check:
        print("Is our contingency matrix the same as sklearn's?\n",
            np.array(joint_probabilities[[str(val) for val in np.unique(b)]].loc[[str(val) for val in np.unique(a)]] * len(a), dtype=int) == cluster.contingency_matrix(a, b))
        print("Do they add up to 1?\n", np.array(joint_probabilities).sum())

    return joint_probabilities


def label_the_data(estimator, A_vector, *args, **kwargs):  # say the estimator is Binarizer
    n_bins = kwargs.get('n_bins', None)

    A_vector = np.array(A_vector) # because out code needs it to be a numpy array

    if len(args) == 1:
        B_vector = np.array(args[0]) # because out code needs it to be a numpy array
        # Fit it using the distribution from both vectors
        estimator.fit(np.concatenate([A_vector, B_vector], axis=0).flatten().reshape(-1, 1))

        # the labeled data
        A_transformed = estimator.transform(A_vector.reshape(-1, 1)).flatten()
        B_transformed = estimator.transform(B_vector.reshape(-1, 1)).flatten()

        # To check it has the number of bins we asked of it
        if n_bins:
            if len(np.concatenate([np.unique(A_transformed), np.unique(B_transformed)], axis=0).flatten()) != n_bins:
                IOError("The number of bins is not the same in label the data {} != {}".format(len(np.concatenate([np.unique(A_transformed), np.unique(B_transformed)], axis=0).flatten()), n_bins))

        # what to give back
        transformed_vectors = [pd.Series(A_transformed), pd.Series(B_transformed)]
    elif len(args) > 1:
        IOError("too many *args in label the data")
    else:
        # Fit it using the distribution from only one vector
        estimator.fit(A_vector.flatten().reshape(-1, 1))

        # The labeled data
        A_transformed = estimator.transform(A_vector.reshape(-1, 1)).flatten()

        # To check it has the number of bins we asked of it
        if n_bins:
            if len(np.unique(A_transformed)) != n_bins:
                IOError("The number of bins is not the same in label the data {} != {}".format(len(np.unique(A_transformed).flatten()), n_bins))

        # what to give back
        transformed_vectors = [pd.Series(A_transformed)]

    return transformed_vectors


def saferound_the_df(df, precision):
    # Here we round the numbers so that they add up to 1
    df = pd.DataFrame(data=np.array(iteround.saferound(np.array(df).flatten(), precision)).reshape(len(df.index), len(df.columns)),
        columns=df.columns, index=df.index)

    return df


def set_estimator(A, *args, **kwargs):
    def set_Binarizer_estimator(A, *args):
        if len(args) > 0:
            B = args[0]
            est = Binarizer(threshold=np.median(np.concatenate([np.array(A), np.array(B)], axis=0)))
        else:
            est = Binarizer(threshold=np.median(np.array(A)))

        return est

    estimator = kwargs.get('estimator', 'Binarizer')

    if estimator == 'Binarizer':
        est = set_Binarizer_estimator(A, *args)
    else:
        n_bins = kwargs.get('n_bins', 3)
        est = KBinsDiscretizer(n_bins=n_bins, strategy=estimator, encode='ordinal')

    return est


def various_datasets_average_joint_probabilities(joint_probs_dictionary, datasets, labels, **kwargs):
    filename = kwargs.get('filename', 'Unknown filename')
    precision = kwargs.get('precision', 2)
    directory = kwargs.get('directory', '')
    cmap = kwargs.get('cmap', plt.get_cmap('magma'))

    all_probs = np.array([np.array(joint_probs_dictionary[dataset][label]).flatten() for label in labels for dataset in datasets]).flatten()
    vmax = np.max(all_probs)
    vmin = np.min(all_probs)

    fig, axes = plt.subplots(len(list(datasets)), len(list(labels)), sharex='col', sharey='row', figsize=(12.7, 7.5))

    sc = axes[0, 0].scatter(range(20), range(20), vmax=vmax, vmin=vmin, cmap=cmap, c=range(20))

    for dataset, row in zip(datasets, range(len(datasets))):
        for label, column in zip(labels, range(len(labels))):
            if joint_probs_dictionary[dataset][label].shape[0]*joint_probs_dictionary[dataset][label].shape[1] > 16:
                annot = False
            else:
                annot = True
            sns.heatmap(joint_probs_dictionary[dataset][label], ax=axes[row, column], annot=annot, vmin=vmin, vmax=vmax, cmap=cmap, cbar=False, fmt='.{}f'.format(precision))

    # set the x and y axes
    for ax, dataset in zip(axes[:, 0], datasets):
        ax.set_ylabel(dataset, rotation=90, size='large')

    for ax, label in zip(axes[-1, :], labels):
        ax.set_xlabel(label)

    plt.tight_layout(rect=(0, 0, 1, .96))
    fig.colorbar(sc, ax=axes, use_gridspec=True) # mpl.cm.ScalarMappable(norm=norm, cmap=cmap), , cmap=cmap, norm=norm
    plt.suptitle(filename)
    # plt.tight_layout(pad=.3, rect=(0, 0, 1, .96))  # rect=(0, 0, 1, .97) rect=(0, 0.03, 1, .97),
    plt.show()
    # plt.savefig(directory + '/' + filename, dpi=300)
    plt.close()



def main():
    # import them
    pickle_in = open("measurements.pickle", "rb")
    measurements = pickle.load(pickle_in)
    pickle_in.close()

    unique_classes = np.array([str(val) for val in np.unique(measurements.all_bacteria['generationtime'])])
    print('unique_classes', unique_classes)

    script_to_get_Intra_and_Pooled_boxplot(measurements, how_many_times_to_resample=50, save_bootstrap_df=False, estimator='kmeans', n_bins=10, var='growth_rate')

    exit()

    estimator = Binarizer(threshold=np.median(np.concatenate([measurements.sis_A_pooled['generationtime'], measurements.sis_B_pooled['generationtime']], axis=0)))
    # estimator = KBinsDiscretizer(n_bins=3, strategy='quantile', encode='ordinal')

    sis_A_label_pooled, sis_B_label_pooled = label_the_data(estimator, np.array(measurements.sis_A_pooled['generationtime']), np.array(measurements.sis_B_pooled['generationtime']))

    original_bootstraped_MI_and_jointprobs_and_ci(pd.Series(sis_A_label_pooled), pd.Series(sis_B_label_pooled), how_many_times_to_resample=50, show_joint_probability=True)

    exit()

    # def

    print(label_the_data(KBinsDiscretizer(n_bins=4, strategy='kmeans', encode='ordinal'), np.array(measurements.sis_A_pooled['generationtime'])))
    exit()

    q1 = np.percentile(measurements.all_bacteria['generationtime'], 25)
    median = np.percentile(measurements.all_bacteria['generationtime'], 50)
    q2 = np.percentile(measurements.all_bacteria['generationtime'], 75)

    sns.distplot(measurements.all_bacteria['generationtime'],
        label="{} {} {} {}".format(len(measurements.all_bacteria[measurements.all_bacteria['generationtime'] <= q1]) / len(measurements.all_bacteria['generationtime']),
                                   len(measurements.all_bacteria[(measurements.all_bacteria['generationtime'] > q1) & (measurements.all_bacteria['generationtime'] <= median)]) / len(
                                       measurements.all_bacteria['generationtime']),
                                   len(measurements.all_bacteria[(measurements.all_bacteria['generationtime'] > median) & (measurements.all_bacteria['generationtime'] <= q2)]) / len(
                                       measurements.all_bacteria['generationtime']),
                                   len(measurements.all_bacteria[(measurements.all_bacteria['generationtime'] > q2)]) / len(measurements.all_bacteria['generationtime']), ))
    plt.axvline(q1)
    plt.axvline(median)
    plt.axvline(q2)
    plt.legend()
    plt.show()
    plt.close()

    qt = QuantileTransformer(n_quantiles=4)
    transformed = qt.fit_transform(np.array(measurements.all_bacteria['generationtime']).reshape(1, -1))
    print(transformed)
    print(qt.n_quantiles_)
    print(qt.quantiles_)
    print(qt.references)
    exit()

    joint_dist_matrices = pd.DataFrame(data=np.zeros((len(unique_classes), len(unique_classes))), columns=unique_classes, index=unique_classes)  # A classes as rows and B classes as columns

    MI(np.array(measurements.sis_A_pooled['generationtime']), np.array(measurements.sis_B_pooled['generationtime']), check=True)
    joint_probability_matrix(np.array(measurements.sis_A_pooled['generationtime']), np.array(measurements.sis_B_pooled['generationtime']), joint_dist_matrices, check=True)

    labels_true = measurements.sis_A_intra_gen_bacteria[0]['generationtime']
    labels_pred = measurements.sis_B_intra_gen_bacteria[0]['generationtime']
    cont_mat = cluster.contingency_matrix(labels_true, labels_pred)
    print(len(labels_pred), len(labels_true))
    print(type(cont_mat))
    print(cont_mat.shape)
    print(cont_mat / len(labels_true))
    print(np.array(cont_mat / len(labels_true)).flatten().sum())
    exit()

    how_many_times_to_resample = 2000
    percent_to_cover = .9
    directory = 'Generationtime Analysis {} resamples, {} percent covered'.format(how_many_times_to_resample, percent_to_cover)
    create_folder(directory)
    save_bootstrap_df = True

    script_to_get_Inter_boxplot(measurements, how_many_times_to_resample, percent_to_cover, directory, save_bootstrap_df)
    exit()

    script_to_get_only_Pooled_boxplot(measurements, how_many_times_to_resample, percent_to_cover, directory)
    print('Finished the Pooled boxplots')

    exit()


if __name__ == '__main__':
    main()
