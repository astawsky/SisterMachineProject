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


def saferound_the_df(df, precision):
    # Here we round the numbers so that they add up to 1
    df = pd.DataFrame(data=np.array(iteround.saferound(np.array(df).flatten(), precision)).reshape(len(df.index), len(df.columns)), columns=df.columns, index=df.index)

    return df


def joint_probability_matrix(a, b, **kwargs):
    check = kwargs.get('check', False)  # this is to check our method against sklearn's
    # precision = kwargs.get('precision', 3)
    # joint_probabilities = kwargs.get('joint_probabilities', 'new')

    joint_probabilities = pd.DataFrame(data=np.zeros((len(np.unique(a)), len(np.unique(b)))), columns=[str(val) for val in np.unique(b)], index=[str(val) for val in np.unique(a)])
    # if joint_probabilities == 'new':
    #     joint_probabilities = pd.DataFrame(data=np.zeros((len(np.unique(a)), len(np.unique(b)))), columns=[str(val) for val in np.unique(b)], index=[str(val) for val in np.unique(a)])

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
    # if np.array(joint_probabilities).sum() != len(together):
    #     IOError("The sum of the joint distribution matrix is not the number of samples {} != {}".format(np.array(joint_probabilities).sum(), len(together)))

    # Debugging
    if check:
        print("Is our contingency matrix the same as sklearn's?\n",
            np.array(joint_probabilities[[str(val) for val in np.unique(b)]].loc[[str(val) for val in np.unique(a)]] * len(a), dtype=int) == cluster.contingency_matrix(a, b))
        print("Do they add up to 1?\n", np.array(joint_probabilities).sum())

    return joint_probabilities


class Bootstrap(object):

    def label_the_data(self, A_vector, *args, **kwargs):  # say the estimator is Binarizer
        def set_estimator(A, *args, **kwargs):
            if self.estimator == 'Binarizer':  # special case because we need the median at the initialization of the estimator
                if args[0] is not None:  # If we input two vectors, use the pooled median of A and B
                    B = args[0]
                    est = Binarizer(threshold=np.median(np.concatenate([np.array(A), np.array(B)], axis=0)))
                else:  # If we input just one vector, use just the median of A
                    est = Binarizer(threshold=np.median(np.array(A)))
            elif self.estimator == 'values':  # using the values as the bins (ONLY WORKS WITH GENERATIONTIME)
                # create a class that has the .fit() and .transform() attribute
                class Values_Discrete(object):
                    def transform(self, samples):
                        return samples

                    def fit(self, samples):
                        pass

                est = Values_Discrete()
            elif self.estimator == 'MI estimator':
                class Estimator(Bootstrap):
                    def __init__(self, **kwargs):
                        self.knn = kwargs.get('knn', 3)

                        # decides the "features" are discrete
                        if self.var1 == 'generationtime':
                            self.discrete_features = True
                        else:
                            self.discrete_features = False

                        # decides which algorithm to use
                        if self.var2 == 'generationtime':
                            self.discrete_target = True
                        else:
                            self.discrete_target = False

                    def fit(self, np_array):
                        self.X = np_array

                    def transform(self, y):
                        self.y = y
                        if self.discrete_target:
                            mutual_info_classif(X=self.X, y=self.y, discrete_features=self.discrete_features, n_neighbors=self.knn)
                        else:
                            mutual_info_regression(X=self.X, y=self.y, discrete_features=self.discrete_features, n_neighbors=self.knn)
            # elif self.estimator == 'pearson correlation':
            #     class Pearson(Bootstrap):
            #         def __init__(self):
            #             pass
            #
            #         def transform(self, ):
            #             np.array
            else:  # uses the KBinsDiscretizer from sklearn and a choice of the 'uniform', 'percentile', or 'kmeans' strategies; 'kmeans' is preferred
                n_bins = kwargs.get('n_bins', 3)
                est = KBinsDiscretizer(n_bins=n_bins, strategy=self.estimator, encode='ordinal')

            return est

        final_index = kwargs.get('max_size', len(A_vector))
        n_bins = kwargs.get('n_bins', None)
        different_variables = kwargs.get('different_variables', False) # calculate the median from the two vectors seperately, ie. have different estimators for both

        random_indices = np.random.choice(len(A_vector), final_index, False) # so there is not a bias

        A_vector = np.array(A_vector)[random_indices]  # because our code needs it to be a numpy array and to trim the vector

        if args[0] is not None:  # If we are labeling
            B_vector = np.array(args[0])[random_indices]  # because out code needs it to be a numpy array
            args = [None]
            if different_variables: # use different estimators for two vectors of different variables
                est_A = set_estimator(A_vector, *args, **kwargs)
                est_B = set_estimator(B_vector, *args, **kwargs)
                # Fit it using the distribution from one vector
                est_A.fit(A_vector.flatten().reshape(-1, 1))
                # Fit it using the distribution from one vector
                est_B.fit(B_vector.flatten().reshape(-1, 1))
            else: # est_A and est_B are the same, ie. we use both vectors to compute the median
                est_A = set_estimator(A_vector, *args, **kwargs)
                # Fit it using the distribution from both vectors
                est_A.fit(np.concatenate([A_vector, B_vector], axis=0).flatten().reshape(-1, 1))
                # est_B is the same
                est_B = est_A

            # the labeled data
            A_transformed = est_A.transform(A_vector.reshape(-1, 1)).flatten()
            B_transformed = est_B.transform(B_vector.reshape(-1, 1)).flatten()

            # To check it has the number of bins we asked of it
            if n_bins:
                if len(np.unique(np.concatenate([A_transformed, B_transformed], axis=0)).flatten()) != n_bins:
                    IOError("The number of bins is not the same in label the data {} != {}".format(len(np.concatenate([np.unique(A_transformed), np.unique(B_transformed)], axis=0).flatten()), n_bins))

            # what to give back
            transformed_vectors = [pd.Series(A_transformed), pd.Series(B_transformed)]
        elif len(args) > 1:
            raise IOError("too many *args in label the data")
        else:
            # get the estimator
            est = set_estimator(A_vector, *args, **kwargs)

            # Fit it using the distribution from only one vector
            est.fit(A_vector.flatten().reshape(-1, 1))

            # The labeled data
            A_transformed = est.transform(A_vector.reshape(-1, 1)).flatten()

            # To check it has the number of bins we asked of it
            if n_bins:
                if len(np.unique(A_transformed)) != n_bins:
                    IOError("The number of bins is not the same in label the data {} != {}".format(len(np.unique(A_transformed).flatten()), n_bins))

            # what to give back
            transformed_vectors = [pd.Series(A_transformed)]


        return transformed_vectors

    # get two bootstrapped lists of A_labeled (and B_labeled) vectors
    def bootstrapping(self):

        return_array = []
        self.A_bootstraps = {}
        self.B_bootstraps = {}

        for resample_number in range(self.how_many_times_to_resample):
            # resampled with replacement data
            A_new = self.label_A.sample(n=len(self.label_A), replace=True, axis='index')
            self.A_bootstraps.update({resample_number: A_new})

            # If we are using two vectors and want to syncronize them
            if not isinstance(self.label_B, type(None)):
                B_new = self.label_B.loc[A_new.index]
                self.B_bootstraps.update({resample_number: B_new})

        # update what to send out, therefore we can change the amount of things we send out
        return_array.append(self.A_bootstraps)
        if self.B_bootstraps:
            return_array.append(self.B_bootstraps)

            # for the joint probability matrix to be understandable, this way it is faster
            self.bin_labels = np.array([str(val) for val in np.unique(np.concatenate([np.array(self.label_A), np.array(self.label_B)], axis=0).flatten())])
            return_array.append(self.bin_labels)

        return return_array

    # get the MIs from the bootstrapped lists
    def get_MIs_from_bootstraps(self):

        # checking that they have the same number of bootstraps
        if self.A_bootstraps.keys() != self.B_bootstraps.keys():
            raise IOError('get_MIs_from_bootstraps shows that A and B bootstraps do not have the same keys: {} and {}'.format(len(self.A_bootstraps), len(self.B_bootstraps)))

        # where we will put the different MIs
        self.mutual_info_dictionary = dict(zip(self.types_of_MI, [[] for blah in range(len(self.types_of_MI))]))

        # number_of_bootstraps = len(A_bootstraps)
        for key in self.A_bootstraps.keys():
            A = self.A_bootstraps[key]
            B = self.B_bootstraps[key]

            # get and save the MIs we want
            self.mutual_info_dictionary['Normalized MI'].append(normalized_mutual_info_score(A, B, average_method='arithmetic'))
            # if 'MI' in self.types_of_MI:
            #     self.mutual_info_dictionary['MI'].append(mutual_info_score(A, B))
            # if 'Normalized MI' in self.types_of_MI:
            #     self.mutual_info_dictionary['Normalized MI'].append(normalized_mutual_info_score(A, B, average_method='arithmetic'))
            # if 'Adjusted MI' in self.types_of_MI:
            #     self.mutual_info_dictionary['Adjusted MI'].append(adjusted_mutual_info_score(A, B, average_method='arithmetic'))



        return self.mutual_info_dictionary

    # # get the joint probability matrix from two labeled vectors
    # def joint_probability_matrix(self, **kwargs):
    #     check = kwargs.get('check', False)  # this is to check our method against sklearn's
    #     joint_probabilities = kwargs.get('joint_probabilities', 'new')
    #
    #     # create a new probability matrix or add to one already given?
    #     if joint_probabilities == 'new':
    #         joint_probabilities = pd.DataFrame(data=np.zeros((len(np.unique(self.label_A)), len(np.unique(self.label_B)))), columns=[str(val) for val in np.unique(self.label_B)],
    #             index=[str(val) for val in np.unique(self.label_A)])
    #
    #     together = pd.DataFrame({'self.label_A': self.label_A, 'self.label_B': self.label_B})  # for the joint distribution
    #     for a_class in np.unique(self.label_A):
    #         for b_class in np.unique(self.label_B):
    #             joint = len(together[(together['label_A'] == a_class) & (together['label_B'] == b_class)])
    #             joint_probabilities[str(b_class)].loc[str(a_class)] = joint_probabilities[str(b_class)].loc[str(a_class)] + joint
    #
    #     # # Here we round the numbers so that they add up to 1
    #     # joint_probabilities = pd.DataFrame(data=np.array([iteround.saferound(np.array(joint_probabilities / len(together))[ind, :], precision) for ind in range(np.array(joint_probabilities).shape[0])]),
    #     #     columns=[str(val) for val in np.unique(self.label_B)], index=[str(val) for val in np.unique(self.label_A)])
    #
    #     # # If the joint distribution matrix does not equal exactly 1 there must be a problem!
    #     # if np.array(joint_probabilities).sum() != 1.0:
    #     #     IOError("The sum of the joint distribution matrix is {} != 1.0".format(np.array(joint_probabilities).sum()))
    #
    #     # If the joint distribution matrix does not equal exactly to the amount of samples there must be a problem!
    #     if np.array(joint_probabilities).sum() != len(together):
    #         IOError("The sum of the joint distribution matrix is not the number of samples {} != {}".format(np.array(joint_probabilities).sum(), len(together)))
    #
    #     # Debugging
    #     if check:
    #         print("Is our contingency matrix the same as sklearn's?\n",
    #             np.array(joint_probabilities[[str(val) for val in np.unique(self.label_B)]].loc[[str(val) for val in np.unique(self.label_A)]] * len(self.label_A),
    #                 dtype=int) == cluster.contingency_matrix(self.label_A, self.label_B))
    #         print("Do they add up to 1?\n", np.array(joint_probabilities).sum())
    #
    #     return joint_probabilities

    # save or show the graph of the heatmap of the joint probability matrix or the average joint probability matrix
    def heatmap_of_joint_prob(self, mean_joint_probs, **kwargs):
        directory = kwargs.get('directory', None)
        save_or_show = kwargs.get('save_or_show', 'save')
        title = 'Joint Probability, {} bootstraps, {} estimator'.format(self.how_many_times_to_resample, self.estimator)
        if self.estimator not in ['values', 'Binarizer']:
            n_bins = kwargs.get('n_bins', 3)
            title = title + ', discretized to {} number of bins'.format(n_bins)
        if np.array(mean_joint_probs).shape[0] * np.array(mean_joint_probs).shape[1] > 25:
            annot = False
        else:
            annot = True
        sns.heatmap(mean_joint_probs, annot=annot, fmt='.3f')
        plt.title(title)
        plt.xlabel('A trajectory bins')
        plt.xlabel('B trajectory bins')
        plt.tight_layout()
        if save_or_show == 'save':
            plt.savefig(directory + "/{}".format(title), dpi=300)
        elif save_or_show == 'show':
            plt.show()
        else:
            raise IOError('heatmap_of_joint_prob: save_or_show != save or show')
        plt.close()

    # Get the average probability matrix from the bootstraps
    def get_joint_prob_matrix_from_bootstraps(self, **kwargs):

        precision = kwargs.get('precision', 2)

        # checking that they have the same number of bootstraps
        if self.A_bootstraps.keys() != self.B_bootstraps.keys():
            raise IOError('get_MIs_from_bootstraps shows that A and B bootstraps do not have the same keys: {} and {}'.format(len(self.A_bootstraps), len(self.B_bootstraps)))

        # where we store all the joint prob matrices from all the bootstraps
        joint_probabilities_array = []

        # number_of_bootstraps = len(self.A_bootstraps)
        for key in self.A_bootstraps.keys():
            A = self.A_bootstraps[key]
            B = self.B_bootstraps[key]

            # calculate the joint probabilities matrix and convert it to dictionary to make it simpler
            joint_probabilities_array.append(joint_probability_matrix(np.array(A), np.array(B), **kwargs).to_dict())

        # we put all the joint probabilities in array order to round them so that they add up to 1 with the least amount of perturbations
        something = np.array([[np.array([joint_probabilities_array[ind][column][row] for ind in range(len(joint_probabilities_array)) if
                                         (column in list(joint_probabilities_array[ind].keys())) and (row in list(joint_probabilities_array[ind][column].keys()))]).sum() for row in self.bin_labels]
                              for column in self.bin_labels]).flatten()

        # we round them and reshape them to a numpy array, then a dictionary and finally to a DataFrame to heatmap correctly
        rounded_flat = np.array(iteround.saferound(something / (self.how_many_times_to_resample * len(list(self.A_bootstraps.values())[0])), precision)).reshape(len(self.bin_labels), len(self.bin_labels))
        self.mean_joint_probs = pd.DataFrame.from_dict(
            {column_label: {row_label: rounded_flat[row, column] for row, row_label in zip(range(len(self.bin_labels)), self.bin_labels)} for column, column_label in
             zip(range(len(self.bin_labels)), self.bin_labels)})

        return self.mean_joint_probs

    # This is another measure of similarity
    def get_pearson_correlation_from_bootstraps(self):
        # checking that they have the same number of bootstraps
        if self.A_bootstraps.keys() != self.B_bootstraps.keys():
            raise IOError('get_MIs_from_bootstraps shows that A and B bootstraps do not have the same keys: {} and {}'.format(len(self.A_bootstraps), len(self.B_bootstraps)))

        # where we will put the different pearson correlations
        self.pearson_dictionary = np.array([stats.pearsonr(self.A_bootstraps[key], self.B_bootstraps[key])[0] for key in self.A_bootstraps.keys()])

        return self.pearson_dictionary
    
    def __init__(self, A_vector, *args, **kwargs):

        if not args:
            B_vector = None
        elif len(args) == 1:
            B_vector = args[0]
        else:
            raise IOError('Too many arguments! Only two 1D arrays, preferrably pd.Series type...')

        self.how_many_times_to_resample = kwargs.get('how_many_times_to_resample', 200)

        self.estimator = kwargs.get('estimator', 'Binarizer')  # what estimator to use to label the data

        self.types_of_MI = kwargs.get("types_of_MI", ['MI', 'Normalized MI', 'Adjusted MI'])  # to see what MIs we want
        
        self.var1, self.var2 = kwargs.get('var1', 'generationtime'), kwargs.get('var2', 'generationtime')

        """ Get the bootstraps """

        self.label_A, self.label_B = self.label_the_data(A_vector, B_vector, **kwargs)

        self.bootstrapping()

def Pearson_boxplots(measurements, **kwargs):
    var1 = kwargs.get('var1', 'generationtime')
    var2 = kwargs.get('var2', 'generationtime')
    directory = kwargs.get('directory', '')
    window_size = kwargs.get('window_size', 3)
    if directory == '':
        pass
    else:
        directory = directory + '/'

    # Intra 0-7, Intra windows, Pooled, Inter are the ones we'll save
    number_of_window_datasets = len(measurements.sis_A_intra_gen_bacteria) - window_size + 1

    sis_A_window = [pd.concat(measurements.sis_A_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_window_datasets)]
    sis_B_window = [pd.concat(measurements.sis_B_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_window_datasets)]
    non_sis_A_window = [pd.concat(measurements.non_sis_A_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_window_datasets)]
    non_sis_B_window = [pd.concat(measurements.non_sis_B_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_window_datasets)]
    con_A_window = [pd.concat(measurements.con_A_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_window_datasets)]
    con_B_window = [pd.concat(measurements.con_B_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_window_datasets)]
    
    sis_A_array = measurements.sis_A_intra_gen_bacteria.copy() + sis_A_window + [measurements.sis_A_pooled]
    sis_B_array = measurements.sis_B_intra_gen_bacteria.copy() + sis_B_window + [measurements.sis_B_pooled]
    non_sis_A_array = measurements.non_sis_A_intra_gen_bacteria.copy() + non_sis_A_window + [measurements.non_sis_A_pooled]
    non_sis_B_array = measurements.non_sis_B_intra_gen_bacteria.copy() + non_sis_B_window + [measurements.non_sis_B_pooled]
    con_A_array = measurements.con_A_intra_gen_bacteria.copy() + con_A_window + [measurements.con_A_pooled]
    con_B_array = measurements.con_B_intra_gen_bacteria.copy() + con_B_window + [measurements.con_B_pooled]
    Intra_labels = ['Intra {}'.format(ind) for ind in range(len(measurements.sis_A_intra_gen_bacteria))]
    Intra_window_labels = ['Intra {}-{}'.format(ind, ind + window_size) for ind in range(number_of_window_datasets)]
    labels = Intra_labels + Intra_window_labels + ['Pooled Traps']
    print('labels are ', labels)
    
    # Here we will store all the Bootstraps
    # intras_windows_pooled_inter_dictionary = dict()

    pearson_corr_df = pd.DataFrame(columns=['pearson correlation', 'dataset', 'label'])

    for sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, label in zip(sis_A_array, sis_B_array, non_sis_A_array, non_sis_B_array, con_A_array, con_B_array, labels):
        final_index = min(len(sis_A), len(sis_B), len(non_sis_A), len(non_sis_B), len(con_A), len(con_B))
        print(label, final_index)
        # intras_windows_pooled_inter_dictionary.update({label: dict()})
        # Initialize the bootstrap instance and get the values estimator bootstraps
        sis = Bootstrap(sis_A[var1], sis_B[var2], **kwargs)
        pearson_corr_df = pearson_corr_df.append(
            pd.DataFrame({'pearson correlation': [stats.pearsonr(sis.A_bootstraps[key], sis.B_bootstraps[key])[0] for key in sis.A_bootstraps.keys()], 'dataset': ['Sisters' for key in sis.A_bootstraps.keys()],
             'label': [label for key in sis.A_bootstraps.keys()]}), ignore_index=True)
        # intras_windows_pooled_inter_dictionary[label].update({'sis': sis})
        # print('  Sisters done')
        non_sis = Bootstrap(non_sis_A[var1], non_sis_B[var2], **kwargs)
        pearson_corr_df = pearson_corr_df.append(
            pd.DataFrame({'pearson correlation': [stats.pearsonr(non_sis.A_bootstraps[key], non_sis.B_bootstraps[key])[0] for key in non_sis.A_bootstraps.keys()], 'dataset': ['Non-Sisters' for key in non_sis.A_bootstraps.keys()],
             'label': [label for key in non_sis.A_bootstraps.keys()]}), ignore_index=True)
        # intras_windows_pooled_inter_dictionary[label].update({'non_sis': non_sis})
        # print('  Non-Sisters done')
        con = Bootstrap(con_A[var1], con_B[var2], **kwargs)
        pearson_corr_df = pearson_corr_df.append(
            pd.DataFrame({'pearson correlation': [stats.pearsonr(con.A_bootstraps[key], con.B_bootstraps[key])[0] for key in con.A_bootstraps.keys()], 'dataset': ['Control' for key in con.A_bootstraps.keys()],
             'label': [label for key in con.A_bootstraps.keys()]}), ignore_index=True)
        # intras_windows_pooled_inter_dictionary[label].update({'con': con})
        # print('  Control done')

    # Intra and Pooled
    sns.boxplot(x="label", y="pearson correlation", hue="dataset", palette="Set3", data=pearson_corr_df.iloc[[label in Intra_labels+['Pooled Traps'] for label in pearson_corr_df['label']]])
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '{} {} pearson Intra and Pooled'.format(var1, var2), dpi=300)
    plt.close()

    # Intra window
    sns.boxplot(x="label", y="pearson correlation", hue="dataset", palette="Set3", data=pearson_corr_df.iloc[[label in Intra_window_labels for label in pearson_corr_df['label']]])
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '{} {} pearson Intra, window size {}'.format(var1, var2, window_size), dpi=300)
    plt.close()

    # Pooled
    sns.boxplot(x="label", y="pearson correlation", hue="dataset", palette="Set3", data=pearson_corr_df.iloc[[label == 'Pooled Traps' for label in pearson_corr_df['label']]])
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '{} {} pearson Pooled'.format(var1, var2), dpi=300)
    plt.close()

    # Since Inter has no Sisters, Non-Sisters and Control, we do a
    mother_array = measurements.mother_dfs.copy()
    daughter_array = measurements.daughter_dfs.copy()
    Inter_labels = ['Inter {}'.format(ind) for ind in range(len(measurements.mother_dfs))]

    for mom, daughter, label in zip(mother_array, daughter_array, Inter_labels):
        print(label)
        inter = Bootstrap(mom[var1], daughter[var2], **kwargs)
        pearson_corr_df = pearson_corr_df.append(pd.DataFrame(
            {'pearson correlation': [stats.pearsonr(inter.A_bootstraps[key], inter.B_bootstraps[key])[0] for key in inter.A_bootstraps.keys()], 'dataset': ['Control' for key in inter.A_bootstraps.keys()],
             'label': [label for key in inter.A_bootstraps.keys()]}), ignore_index=True)
        # intras_windows_pooled_inter_dictionary.update({label: inter})
        # print('  Inter done')

    # Inter
    sns.boxplot(x="label", y="pearson correlation", palette="Set3", data=pearson_corr_df.iloc[[label in Inter_labels for label in pearson_corr_df['label']]])
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '{} {} pearson Inter'.format(var1, var2), dpi=300)
    plt.close()


    # # save bootstrap df so we don't have to go through that all again
    # pickle_out = open(directory + "{} {} values estimator intras_windows_pooled_inter_dictionary.pickle".format(var1, var2), "wb")
    # pickle.dump(intras_windows_pooled_inter_dictionary, pickle_out)
    # pickle_out.close()


def Intra_and_Pooled_boxplot(measurements, save_bootstrap_df, **kwargs):
    var1 = kwargs.get('var1', 'generationtime')
    var2 = kwargs.get('var2', 'generationtime')
    directory = kwargs.get('directory', '')
    if directory == '':
        pass
    else:
        directory = directory + '/'

    types_of_MI = kwargs.get('types_of_MI', ['NMI'])
    bootstrap_df = pd.DataFrame(columns=types_of_MI + ['dataset', 'label'])
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
        final_index = min(len(sis_A), len(sis_B), len(non_sis_A), len(non_sis_B), len(con_A), len(con_B))
        print(label, final_index)
        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        sis = Bootstrap(sis_A[var1], sis_B[var2], **kwargs)
        sis.get_MIs_from_bootstraps()
        sis.get_joint_prob_matrix_from_bootstraps(**kwargs)
        
        sis.mutual_info_dictionary.update({'dataset': ['Sisters' for ind in range(sis.how_many_times_to_resample)], 'label': [label for ind in range(sis.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=sis.mutual_info_dictionary)], axis=0).reset_index(drop=True)
        joint_probs_dictionary['Sisters'].update({label: sis.mean_joint_probs})

        print('  Sisters done')

        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        non_sis = Bootstrap(non_sis_A[var1], non_sis_B[var2], **kwargs)
        non_sis.get_MIs_from_bootstraps()
        non_sis.get_joint_prob_matrix_from_bootstraps(**kwargs)

        non_sis.mutual_info_dictionary.update({'dataset': ['Non-Sisters' for ind in range(non_sis.how_many_times_to_resample)], 'label': [label for ind in range(non_sis.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=non_sis.mutual_info_dictionary)], axis=0).reset_index(drop=True)
        joint_probs_dictionary['Non-Sisters'].update({label: non_sis.mean_joint_probs})

        print('  Non-Sisters done')

        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        con = Bootstrap(con_A[var1], con_B[var2], **kwargs)
        con.get_MIs_from_bootstraps()
        con.get_joint_prob_matrix_from_bootstraps(**kwargs)

        con.mutual_info_dictionary.update({'dataset': ['Control' for ind in range(con.how_many_times_to_resample)], 'label': [label for ind in range(con.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=con.mutual_info_dictionary)], axis=0).reset_index(drop=True)
        joint_probs_dictionary['Control'].update({label: con.mean_joint_probs})

        print('  Control done')

    if save_bootstrap_df:
        # save bootstrap df so we don't have to go through that all again
        pickle_out = open(directory+"Intra and Pooled bootstrap_df.pickle", "wb")
        pickle.dump(bootstrap_df, pickle_out)
        pickle_out.close()
        pickle_out = open(directory + "Intra and Pooled joint_probs_dictionary.pickle", "wb")
        pickle.dump(joint_probs_dictionary, pickle_out)
        pickle_out.close()

    # joint probability distributions
    various_datasets_average_joint_probabilities(joint_probs_dictionary, ['Sisters', 'Non-Sisters', 'Control'], labels, filename='{} {} Intra Average Joint Probabilities'.format(var1, var2), **kwargs)

    # get the Box plots of Intra and Pooled
    if 'MI' in types_of_MI:
        sns.boxplot(x="label", y="MI", hue="dataset", palette="Set3", data=bootstrap_df)
        plt.xticks(rotation=0)
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '{} {} MI Intra and Pooled'.format(var1, var2), dpi=300)
        plt.close()
    if 'Normalized MI' in types_of_MI:
        sns.boxplot(x="label", y="Normalized MI", hue="dataset", palette="Set3", data=bootstrap_df)
        plt.xticks(rotation=0)
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '{} {} NMI Intra and Pooled'.format(var1, var2), dpi=300)
        plt.close()
    if 'Adjusted MI' in types_of_MI:
        sns.boxplot(x="label", y="Adjusted MI", hue="dataset", palette="Set3", data=bootstrap_df)
        plt.xticks(rotation=0)
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '{} {} AMI Intra and Pooled'.format(var1, var2), dpi=300)
        plt.close()

    # get the boxplots of only Pooled
    if 'MI' in types_of_MI:
        sns.boxplot(x="label", y="MI", hue="dataset", palette="Set3", data=bootstrap_df[bootstrap_df['label'] == 'Pooled Traps'])
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '{} {} MI Pooled'.format(var1, var2), dpi=300)
        plt.close()
    if 'Normalized MI' in types_of_MI:
        sns.boxplot(x="label", y="Normalized MI", hue="dataset", palette="Set3", data=bootstrap_df[bootstrap_df['label'] == 'Pooled Traps'])
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '{} {} NMI Pooled'.format(var1, var2), dpi=300)
        plt.close()
    if 'Adjusted MI' in types_of_MI:
        sns.boxplot(x="label", y="Adjusted MI", hue="dataset", palette="Set3", data=bootstrap_df[bootstrap_df['label'] == 'Pooled Traps'])
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '{} {} AMI Pooled'.format(var1, var2), dpi=300)
        plt.close()


def Intra_window_boxplot(measurements, save_bootstrap_df, **kwargs):
    var1 = kwargs.get('var1', 'generationtime')
    var2 = kwargs.get('var2', 'generationtime')
    window_size = kwargs.get('window_size', 3)
    directory = kwargs.get('directory', '')
    if directory == '':
        pass
    else:
        directory = directory + '/'

    types_of_MI = kwargs.get('types_of_MI')
    bootstrap_df = pd.DataFrame(columns=types_of_MI + ['dataset', 'label'])
    joint_probs_dictionary = {'Sisters': {}, 'Non-Sisters': {}, 'Control': {}}

    number_of_datasets = len(measurements.sis_A_intra_gen_bacteria) - window_size + 1

    sis_A_array = [pd.concat(measurements.sis_A_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_datasets)]
    sis_B_array = [pd.concat(measurements.sis_B_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_datasets)]
    non_sis_A_array = [pd.concat(measurements.non_sis_A_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_datasets)]
    non_sis_B_array = [pd.concat(measurements.non_sis_B_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_datasets)]
    con_A_array = [pd.concat(measurements.con_A_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_datasets)]
    con_B_array = [pd.concat(measurements.con_B_intra_gen_bacteria.copy()[start:start + window_size], axis=0).reset_index(drop=True) for start in range(number_of_datasets)]
    labels = ['Intra {}-{}'.format(ind, ind+window_size) for ind in range(number_of_datasets)]

    for sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, label in zip(sis_A_array, sis_B_array, non_sis_A_array, non_sis_B_array, con_A_array, con_B_array, labels):
        final_index = min(len(sis_A), len(sis_B), len(non_sis_A), len(non_sis_B), len(con_A), len(con_B))
        print(label, final_index)
        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        sis = Bootstrap(sis_A[var1], sis_B[var2], **kwargs)
        sis.get_MIs_from_bootstraps()
        sis.get_joint_prob_matrix_from_bootstraps(**kwargs)

        sis.mutual_info_dictionary.update({'dataset': ['Sisters' for ind in range(sis.how_many_times_to_resample)], 'label': [label for ind in range(sis.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=sis.mutual_info_dictionary)], axis=0).reset_index(drop=True)
        joint_probs_dictionary['Sisters'].update({label: sis.mean_joint_probs})

        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        non_sis = Bootstrap(non_sis_A[var1], non_sis_B[var2], **kwargs)
        non_sis.get_MIs_from_bootstraps()
        non_sis.get_joint_prob_matrix_from_bootstraps(**kwargs)

        non_sis.mutual_info_dictionary.update({'dataset': ['Non-Sisters' for ind in range(non_sis.how_many_times_to_resample)], 'label': [label for ind in range(non_sis.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=non_sis.mutual_info_dictionary)], axis=0).reset_index(drop=True)
        joint_probs_dictionary['Non-Sisters'].update({label: non_sis.mean_joint_probs})

        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        con = Bootstrap(con_A[var1], con_B[var2], **kwargs)
        con.get_MIs_from_bootstraps()
        con.get_joint_prob_matrix_from_bootstraps(**kwargs)

        con.mutual_info_dictionary.update({'dataset': ['Control' for ind in range(con.how_many_times_to_resample)], 'label': [label for ind in range(con.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=con.mutual_info_dictionary)], axis=0).reset_index(drop=True)
        joint_probs_dictionary['Control'].update({label: con.mean_joint_probs})

    if save_bootstrap_df:
        # save bootstrap df so we don't have to go through that all again
        pickle_out = open(directory + "Intra Window bootstrap_df.pickle", "wb")
        pickle.dump(bootstrap_df, pickle_out)
        pickle_out.close()
        pickle_out = open(directory + "Intra Window joint_probs_dictionary.pickle", "wb")
        pickle.dump(joint_probs_dictionary, pickle_out)
        pickle_out.close()

    # joint probability distributions
    various_datasets_average_joint_probabilities(joint_probs_dictionary, ['Sisters', 'Non-Sisters', 'Control'], labels,
        filename='{} {}, Intra of window size {}, Average Joint Probabilities'.format(var1, var2, window_size), **kwargs)

    # # get the Box plots of Intra and Pooled
    # sns.boxplot(x="label", y="MI", hue="dataset", palette="Set3", data=bootstrap_df)
    # plt.xticks(rotation=0)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(directory + '{} {}, MI Intra of window size {}'.format(var1, var2, window_size), dpi=300)
    # plt.close()

    sns.boxplot(x="label", y="Normalized MI", hue="dataset", palette="Set3", data=bootstrap_df)
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '{} {}, NMI Intra of window size {}'.format(var1, var2, window_size), dpi=300)
    plt.close()

    # sns.boxplot(x="label", y="Adjusted MI", hue="dataset", palette="Set3", data=bootstrap_df)
    # plt.xticks(rotation=0)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(directory + '{} {}, AMI Intra of window size {}'.format(var1, var2, window_size), dpi=300)
    # plt.close()


def script_to_get_Inter_boxplot(measurements, save_bootstrap_df, **kwargs):
    var1 = kwargs.get('var1', 'generationtime')
    var2 = kwargs.get('var2', 'generationtime')
    directory = kwargs.get('directory', '')
    if directory == '':
        pass
    else:
        directory = directory + '/'

    types_of_MI = kwargs.get('types_of_MI')
    bootstrap_df = pd.DataFrame(columns=types_of_MI + ['dataset', 'label'])
    mother_array = measurements.mother_dfs.copy()
    daughter_array = measurements.daughter_dfs.copy()
    labels = ['Inter {}'.format(ind) for ind in range(len(measurements.mother_dfs))]

    for mom, daughter, label in zip(mother_array, daughter_array, labels):
        print(label)
        inter = Bootstrap(mom[var1], daughter[var2], **kwargs)
        inter.get_MIs_from_bootstraps()
        inter.get_joint_prob_matrix_from_bootstraps(**kwargs)

        inter.mutual_info_dictionary.update({'label': [label for ind in range(inter.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=inter.mutual_info_dictionary)], axis=0).reset_index(drop=True)

    if save_bootstrap_df:
        # save bootstrap df so we don't have to go through that all again
        pickle_out = open(directory + "Inter bootstrap_df.pickle", "wb")
        pickle.dump(bootstrap_df, pickle_out)
        pickle_out.close()

    # sns.boxplot(x="label", y="MI", palette="Set3", data=bootstrap_df)
    # plt.xticks(rotation=0)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(directory + '{} {} MI Inter'.format(var1, var2), dpi=300)
    # plt.close()

    sns.boxplot(x="label", y="Normalized MI", palette="Set3", data=bootstrap_df)
    plt.xticks(rotation=30)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '{} {} NMI Inter'.format(var1, var2), dpi=300)
    plt.close()

    # sns.boxplot(x="label", y="Adjusted MI", palette="Set3", data=bootstrap_df)
    # plt.xticks(rotation=0)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(directory + '{} {} AMI Inter'.format(var1, var2), dpi=300)
    # plt.close()


def various_datasets_average_joint_probabilities(joint_probs_dictionary, datasets, labels, **kwargs):
    filename = kwargs.get('filename', 'Unknown filename')
    precision = kwargs.get('precision', 2)
    directory = kwargs.get('directory', '')
    cmap = kwargs.get('cmap', plt.get_cmap('magma'))
    if directory == '':
        pass
    else:
        directory = directory + '/'

    all_probs = np.array([np.array(joint_probs_dictionary[dataset][label]).flatten() for label in labels for dataset in datasets]).flatten()
    vmax = np.max(all_probs)
    vmin = np.min(all_probs)

    fig, axes = plt.subplots(len(list(datasets)), len(list(labels)), sharex='col', sharey='row', figsize=(12.7, 7.5))

    sc = axes[0, 0].scatter(range(20), range(20), vmax=vmax, vmin=vmin, cmap=cmap, c=range(20))

    for dataset, row in zip(datasets, range(len(datasets))):
        for label, column in zip(labels, range(len(labels))):
            if joint_probs_dictionary[dataset][label].shape[0] * joint_probs_dictionary[dataset][label].shape[1] > 16:
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
    fig.colorbar(sc, ax=axes, use_gridspec=True)  # mpl.cm.ScalarMappable(norm=norm, cmap=cmap), , cmap=cmap, norm=norm
    plt.suptitle(filename)
    # plt.tight_layout(pad=.3, rect=(0, 0, 1, .96))  # rect=(0, 0, 1, .97) rect=(0, 0.03, 1, .97),
    # plt.show()
    plt.savefig(directory + filename, dpi=300)
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


def get_folder_of_graphs(measurements, **kwargs):

    Intra_and_Pooled_boxplot(measurements, **kwargs)

    Intra_window_boxplot(measurements, **kwargs)

    script_to_get_Inter_boxplot(measurements, **kwargs)


def pooled_taking_out_datasets(measurements, **kwargs):
    # Here the first tick mark in the x axis is the similarity measure with the pooled traps dataset, the second is the same but with the pooled dataset not containing 0-th generation pairs, then not
    # containing 0,1-th generaitonpairs and so on...

    """ Creates pairs of dataframes of intergenerational relations from all the A/B pairs we find in the subclass's A_dict
                and B_dict. This method is only meant for Sister, Nonsister and Control subclasses; NOT Population. """

    def intragenerational_dataframe_creations(A_dict, B_dict, variable_names, gens_to_take_out):

        A_df_array = []
        B_df_array = []

        for generation in range(gens_to_take_out + 1):

            A_df = pd.DataFrame(columns=variable_names)
            B_df = pd.DataFrame(columns=variable_names)

            # Because it is not a given that all the experiments done will have a this many generations recorded
            A_keys_with_this_length = [keyA for keyA, keyB in zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[keyA]), len(B_dict[keyB])) > generation + 1]
            B_keys_with_this_length = [keyB for keyA, keyB in zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[keyA]), len(B_dict[keyB])) > generation + 1]

            # looping over all pairs recorded, ie. all traps/experiments
            for keyA, keyB in zip(A_keys_with_this_length, B_keys_with_this_length):
                # the indices from each dataframe to include
                index_array = np.array([index for index in np.arange(min(len(A_dict[keyA]), len(B_dict[keyB]))) if index not in np.arange(generation)])

                # add the data, trap means and traj means to the dataframe that collects them for all traps/experiments
                A_df = pd.concat([A_df, A_dict[keyA].iloc[index_array]], axis=0)
                B_df = pd.concat([B_df, B_dict[keyB].iloc[index_array]], axis=0)

            # reset the index because it is convinient and they were all added with an index 0 since we are comparing sisters
            A_df = A_df.reset_index(drop=True)
            B_df = B_df.reset_index(drop=True)

            # add it to the array that contains the sisters, first cousins, second cousins, etc...
            A_df_array.append(A_df)
            B_df_array.append(B_df)

        return A_df_array, B_df_array

    var1 = kwargs.get('var1', 'generationtime')
    var2 = kwargs.get('var2', 'generationtime')
    directory = kwargs.get('directory', '')
    if directory == '':
        pass
    else:
        directory = directory + '/'
    gens_to_take_out = kwargs.get('gens_to_take_out', 7)

    sis_A_array, sis_B_array = intragenerational_dataframe_creations(measurements.sis_A, measurements.sis_B, measurements._variable_names, gens_to_take_out)
    non_sis_A_array, non_sis_B_array = intragenerational_dataframe_creations(measurements.non_sis_A, measurements.non_sis_B, measurements._variable_names, gens_to_take_out)
    con_A_array, con_B_array = intragenerational_dataframe_creations(measurements.con_A, measurements.con_B, measurements._variable_names, gens_to_take_out)

    # for ind in range(len(A_countdown)):
    #     print(ind)
    #     A_countdown[ind]['dataset'] = ['Sisters' for ind1 in range(len(A_countdown[ind]))]
    #     A_countdown[ind]['label'] = ['All generations' if ind==0 else 'excluding gens till {}'.format(ind) for ind1 in range(len(A_countdown[ind]))]
    #     print(A_countdown[ind])
    #
    # exit()

    types_of_MI = kwargs.get('types_of_MI', ['Normalized MI'])
    bootstrap_df = pd.DataFrame(columns=types_of_MI + ['dataset', 'label'])
    # joint_probs_dictionary = {'Sisters': {}, 'Non-Sisters': {}, 'Control': {}}
    # labels = ['All generations' if ind == 0 else 'excluding gens till {}'.format(ind) for ind in range(gens_to_take_out + 1)]
    labels = np.arange(gens_to_take_out + 1)

    final_index = min([min(len(sis_A), len(sis_B), len(non_sis_A), len(non_sis_B), len(con_A), len(con_B)) for sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B in
                       zip(sis_A_array, sis_B_array, non_sis_A_array, non_sis_B_array, con_A_array, con_B_array)])
    print('final_index', final_index)

    sns.set_style("whitegrid")
    plt.plot(np.arange(gens_to_take_out + 1), [min(len(sis_A), len(sis_B)) for sis_A, sis_B in zip(sis_A_array, sis_B_array)], marker='.', color='blue')
    plt.plot(np.arange(gens_to_take_out + 1), [min(len(non_sis_A), len(non_sis_B)) for non_sis_A, non_sis_B in zip(non_sis_A_array, non_sis_B_array)], marker='.', color='orange')
    plt.plot(np.arange(gens_to_take_out + 1), [min(len(con_A), len(con_B)) for con_A, con_B in zip(con_A_array, con_B_array)], marker='.', color='green')
    plt.title('Size of each Pooled till gen dataset (min of A and B trace)')
    plt.tight_layout()
    plt.savefig(directory + '{} {} decrease of Pooled till gen'.format(var1, var2), dpi=300)
    plt.close()

    # pearson_corr_df = pd.DataFrame(columns=['pearson correlation', 'dataset', 'label'])

    for sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, label in zip(sis_A_array, sis_B_array, non_sis_A_array, non_sis_B_array, con_A_array, con_B_array, labels):
        # final_index = min(len(sis_A), len(sis_B), len(non_sis_A), len(non_sis_B), len(con_A), len(con_B))
        print(label)
        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        sis = Bootstrap(sis_A[var1], sis_B[var2], max_size=final_index, **kwargs) # max_size=final_index,
        sis.get_MIs_from_bootstraps()
        # sis.get_joint_prob_matrix_from_bootstraps(**kwargs)

        sis.mutual_info_dictionary.update({'dataset': ['Sisters' for ind in range(sis.how_many_times_to_resample)], 'label': [label for ind in range(sis.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=sis.mutual_info_dictionary)], axis=0).reset_index(drop=True)
        # joint_probs_dictionary['Sisters'].update({label: sis.mean_joint_probs})

        # sis = Bootstrap(sis_A[var1], sis_B[var2], estimator='values', **kwargs)
        # pearson_corr_df = pearson_corr_df.append(pd.DataFrame(
        #     {'pearson correlation': [stats.pearsonr(sis.A_bootstraps[key], sis.B_bootstraps[key])[0] for key in sis.A_bootstraps.keys()], 'dataset': ['Sisters' for key in sis.A_bootstraps.keys()],
        #      'label': [label for key in sis.A_bootstraps.keys()]}), ignore_index=True)

        print('  Sisters done')

        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        non_sis = Bootstrap(non_sis_A[var1], non_sis_B[var2], max_size=final_index, **kwargs)
        non_sis.get_MIs_from_bootstraps()
        # non_sis.get_joint_prob_matrix_from_bootstraps(**kwargs)

        non_sis.mutual_info_dictionary.update({'dataset': ['Non-Sisters' for ind in range(non_sis.how_many_times_to_resample)], 'label': [label for ind in range(non_sis.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=non_sis.mutual_info_dictionary)], axis=0).reset_index(drop=True)
        # joint_probs_dictionary['Non-Sisters'].update({label: non_sis.mean_joint_probs})

        # non_sis = Bootstrap(non_sis_A[var1], non_sis_B[var2], estimator='values', **kwargs)
        # pearson_corr_df = pearson_corr_df.append(pd.DataFrame({'pearson correlation': [stats.pearsonr(non_sis.A_bootstraps[key], non_sis.B_bootstraps[key])[0] for key in non_sis.A_bootstraps.keys()],
        #                                                        'dataset': ['Non-Sisters' for key in non_sis.A_bootstraps.keys()], 'label': [label for key in non_sis.A_bootstraps.keys()]}),
        #     ignore_index=True)

        print('  Non-Sisters done')

        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        con = Bootstrap(con_A[var1], con_B[var2], max_size=final_index, **kwargs)
        con.get_MIs_from_bootstraps()
        # con.get_joint_prob_matrix_from_bootstraps(**kwargs)

        con.mutual_info_dictionary.update({'dataset': ['Control' for ind in range(con.how_many_times_to_resample)], 'label': [label for ind in range(con.how_many_times_to_resample)]})
        bootstrap_df = pd.concat([bootstrap_df, pd.DataFrame(data=con.mutual_info_dictionary)], axis=0).reset_index(drop=True)
        # joint_probs_dictionary['Control'].update({label: con.mean_joint_probs})

        # con = Bootstrap(con_A[var1], con_B[var2], estimator='values', **kwargs)
        # pearson_corr_df = pearson_corr_df.append(pd.DataFrame(
        #     {'pearson correlation': [stats.pearsonr(con.A_bootstraps[key], con.B_bootstraps[key])[0] for key in con.A_bootstraps.keys()], 'dataset': ['Control' for key in con.A_bootstraps.keys()],
        #      'label': [label for key in con.A_bootstraps.keys()]}), ignore_index=True)

        print('  Control done')

    # # joint probability distributions
    # various_datasets_average_joint_probabilities(joint_probs_dictionary, ['Sisters', 'Non-Sisters', 'Control'], labels, filename='{} {} Pooled till gen Average Joint Probabilities'.format(var1, var2), **kwargs)

    # # Pearson Pooled till gen
    # sns.boxplot(x="label", y="pearson correlation", hue="dataset", palette="Set3", data=pearson_corr_df)
    # plt.xticks(rotation=0)
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig(directory + '{} {} pearson Pooled till gen'.format(var1, var2), dpi=300)
    # plt.close()

    # get the Box plots of Pooled till gen
    if 'MI' in types_of_MI:
        sns.boxplot(x="label", y="MI", hue="dataset", palette="Set3", data=bootstrap_df)
        plt.xticks(rotation=0)
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '{} {} MI Pooled till gen'.format(var1, var2), dpi=300)
        plt.close()
    if 'Normalized MI' in types_of_MI:
        sns.boxplot(x="label", y="Normalized MI", hue="dataset", palette="Set3", data=bootstrap_df)
        plt.xticks(rotation=0)
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + 'New {} {} NMI Pooled till gen'.format(var1, var2), dpi=300)
        plt.close()
    if 'Adjusted MI' in types_of_MI:
        sns.boxplot(x="label", y="Adjusted MI", hue="dataset", palette="Set3", data=bootstrap_df)
        plt.xticks(rotation=0)
        plt.tight_layout()
        # plt.show()
        plt.savefig(directory + '{} {} AMI Pooled till gen'.format(var1, var2), dpi=300)
        plt.close()

def Pearson_pooled_till_gen(measurements, **kwargs):
    # Here the first tick mark in the x axis is the similarity measure with the pooled traps dataset, the second is the same but with the pooled dataset not containing 0-th generation pairs, then not
    # containing 0,1-th generaitonpairs and so on...

    """ Creates pairs of dataframes of intergenerational relations from all the A/B pairs we find in the subclass's A_dict
                and B_dict. This method is only meant for Sister, Nonsister and Control subclasses; NOT Population. """

    def intragenerational_dataframe_creations(A_dict, B_dict, variable_names, gens_to_take_out):

        A_df_array = []
        B_df_array = []

        for generation in range(gens_to_take_out + 1):

            A_df = pd.DataFrame(columns=variable_names)
            B_df = pd.DataFrame(columns=variable_names)

            # Because it is not a given that all the experiments done will have a this many generations recorded
            A_keys_with_this_length = [keyA for keyA, keyB in zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[keyA]), len(B_dict[keyB])) > generation + 1]
            B_keys_with_this_length = [keyB for keyA, keyB in zip(A_dict.keys(), B_dict.keys()) if min(len(A_dict[keyA]), len(B_dict[keyB])) > generation + 1]

            # looping over all pairs recorded, ie. all traps/experiments
            for keyA, keyB in zip(A_keys_with_this_length, B_keys_with_this_length):
                # the indices from each dataframe to include
                index_array = np.array([index for index in np.arange(min(len(A_dict[keyA]), len(B_dict[keyB]))) if index not in np.arange(generation)])

                # add the data, trap means and traj means to the dataframe that collects them for all traps/experiments
                A_df = pd.concat([A_df, A_dict[keyA].iloc[index_array]], axis=0)
                B_df = pd.concat([B_df, B_dict[keyB].iloc[index_array]], axis=0)

            # reset the index because it is convinient and they were all added with an index 0 since we are comparing sisters
            A_df = A_df.reset_index(drop=True)
            B_df = B_df.reset_index(drop=True)

            # add it to the array that contains the sisters, first cousins, second cousins, etc...
            A_df_array.append(A_df)
            B_df_array.append(B_df)

        return A_df_array, B_df_array

    var1 = kwargs.get('var1', 'generationtime')
    var2 = kwargs.get('var2', 'generationtime')
    directory = kwargs.get('directory', '')
    if directory == '':
        pass
    else:
        directory = directory + '/'
    gens_to_take_out = kwargs.get('gens_to_take_out', 7)

    sis_A_array, sis_B_array = intragenerational_dataframe_creations(measurements.sis_A, measurements.sis_B, measurements._variable_names, gens_to_take_out)
    non_sis_A_array, non_sis_B_array = intragenerational_dataframe_creations(measurements.non_sis_A, measurements.non_sis_B, measurements._variable_names, gens_to_take_out)
    con_A_array, con_B_array = intragenerational_dataframe_creations(measurements.con_A, measurements.con_B, measurements._variable_names, gens_to_take_out)

    labels = np.arange(gens_to_take_out + 1)

    final_index = min([min(len(sis_A), len(sis_B), len(non_sis_A), len(non_sis_B), len(con_A), len(con_B)) for sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B in
                       zip(sis_A_array, sis_B_array, non_sis_A_array, non_sis_B_array, con_A_array, con_B_array)])
    print('final_index', final_index)

    pearson_corr_df = pd.DataFrame(columns=['pearson correlation', 'dataset', 'label'])

    for sis_A, sis_B, non_sis_A, non_sis_B, con_A, con_B, label in zip(sis_A_array, sis_B_array, non_sis_A_array, non_sis_B_array, con_A_array, con_B_array, labels):
        # final_index = min(len(sis_A), len(sis_B), len(non_sis_A), len(non_sis_B), len(con_A), len(con_B))
        print(label)
        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        sis = Bootstrap(sis_A[var1], sis_B[var2], **kwargs)  # max_size=final_index,
        sis.get_MIs_from_bootstraps()
        pearson_corr_df = pearson_corr_df.append(pd.DataFrame(
            {'pearson correlation': [stats.pearsonr(sis.A_bootstraps[key], sis.B_bootstraps[key])[0] for key in sis.A_bootstraps.keys()], 'dataset': ['Sisters' for key in sis.A_bootstraps.keys()],
             'label': [label for key in sis.A_bootstraps.keys()]}), ignore_index=True)

        print('  Sisters done')

        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        non_sis = Bootstrap(non_sis_A[var1], non_sis_B[var2], **kwargs)
        non_sis.get_MIs_from_bootstraps()
        pearson_corr_df = pearson_corr_df.append(pd.DataFrame({'pearson correlation': [stats.pearsonr(non_sis.A_bootstraps[key], non_sis.B_bootstraps[key])[0] for key in non_sis.A_bootstraps.keys()],
                                                               'dataset': ['Non-Sisters' for key in non_sis.A_bootstraps.keys()], 'label': [label for key in non_sis.A_bootstraps.keys()]}),
            ignore_index=True)

        print('  Non-Sisters done')

        # Initialize the bootstrap instance and get the MIs and average joint prob matrix
        con = Bootstrap(con_A[var1], con_B[var2], **kwargs)
        con.get_MIs_from_bootstraps()

        pearson_corr_df = pearson_corr_df.append(pd.DataFrame(
            {'pearson correlation': [stats.pearsonr(con.A_bootstraps[key], con.B_bootstraps[key])[0] for key in con.A_bootstraps.keys()], 'dataset': ['Control' for key in con.A_bootstraps.keys()],
             'label': [label for key in con.A_bootstraps.keys()]}), ignore_index=True)

        print('  Control done')

    # # joint probability distributions
    # various_datasets_average_joint_probabilities(joint_probs_dictionary, ['Sisters', 'Non-Sisters', 'Control'], labels, filename='{} {} Pooled till gen Average Joint Probabilities'.format(var1, var2), **kwargs)

    # Pearson Pooled till gen
    sns.boxplot(x="label", y="pearson correlation", hue="dataset", palette="Set3", data=pearson_corr_df)
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '{} {} pearson Pooled till gen'.format(var1, var2), dpi=300)
    plt.close()


def main():
    # import them
    pickle_in = open("measurements.pickle", "rb")
    measurements = pickle.load(pickle_in)
    pickle_in.close()

    for var in ['generationtime', 'growth_rate', 'length_birth', 'division_ratio', 'added_length']:
        # pooled_taking_out_datasets(measurements, estimator='Binarizer', gens_to_take_out=25, var1=var, var2=var, types_of_MI=['Normalized MI'], how_many_times_to_resample=200)
        Pearson_pooled_till_gen(measurements, estimator='values', gens_to_take_out=25, var1=var, var2=var, types_of_MI=['Normalized MI'], how_many_times_to_resample=200)

    exit()

    Pearson_boxplots(measurements, var1='added_length', var2='added_length', estimator='values', how_many_times_to_resample=1000)

    exit()

    directory = 'New {} {}, {} resamples, {} estimator'.format('growth_rate', 'growth_rate', 1000, 'Binarizer')

    pickle_in = open(directory + '/' + "Inter bootstrap_df.pickle", "rb")
    bootstrap_df = pickle.load(pickle_in)
    pickle_in.close()
    
    sns.boxplot(x="label", y="Normalized MI", palette="Set3", data=bootstrap_df)
    plt.xticks(rotation=30)
    plt.tight_layout()
    # plt.show()
    plt.savefig(directory + '/' + '{} {} NMI Inter'.format('growth_rate', 'growth_rate'), dpi=300)
    plt.close()



if __name__ == '__main__':
    main()
