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
import os


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

    # os.mkdir("Population data")
    # os.mkdir("Population data/Mother and daughter data")
    #
    # # the population statistics
    # Population.pop_stats.to_csv("Population data/Population Statistics.csv")
    #
    # # coefficients of variation
    # Population.coeffs_of_vars.to_csv("Population data/Population Coefficients of variation.csv")
    #
    # for ind in range(len(Population.mother_dfs)):
    #     # the mother and daughter dataframes
    #     Population.mother_dfs[ind].to_csv("Population data/Mother and daughter data/Mother with separation "+str(ind)+".csv")
    #     Population.daughter_dfs[ind].to_csv("Population data/Mother and daughter data/Daughter with separation " + str(ind)+".csv")
    #
    # os.mkdir("Control dataset data")
    # os.mkdir("Control dataset data/Protein data")
    # os.mkdir("Control dataset data/Generation data")
    # os.mkdir("Control dataset data/Intra-generationional data")
    # os.mkdir("Control dataset data/Trace Coefficients of variation data")
    # os.mkdir("Control dataset data/Trap Coefficients of variation data")
    # os.mkdir("Control dataset data/Trace Statistics data")
    # os.mkdir("Control dataset data/Trap Statistics data")
    #
    # # Now for Control!
    # # protein
    # for key, val in Control.reference_A_raw_protein_data_dict.items():
    #     Control.A_raw_protein_data_dict[key].to_csv("Control dataset data/Protein data/A trace "+str(key)+", reference "+str(val)+".csv")
    #
    # for key, val in Control.reference_B_raw_protein_data_dict.items():
    #     Control.B_raw_protein_data_dict[key].to_csv("Control dataset data/Protein data/B trace "+str(key)+", reference "+str(val)+".csv")
    #
    # # generation data
    # for key, val in Control.reference_A_dict.items():
    #     Control.A_dict[key].to_csv("Control dataset data/Generation data/A trace "+str(key)+", reference "+str(val)+".csv")
    #
    # for key, val in Control.reference_B_dict.items():
    #     Control.B_dict[key].to_csv("Control dataset data/Generation data/B trace "+str(key)+", reference "+str(val)+".csv")
    #
    # # trace coefficients of variation data
    # for key, val in Control.A_coeffs_of_vars_dict.items():
    #     val.to_csv("Control dataset data/Trace Coefficients of variation data/A trace "+str(key)+".csv")
    #
    # for key, val in Control.B_coeffs_of_vars_dict.items():
    #     val.to_csv("Control dataset data/Trace Coefficients of variation data/B trace "+str(key)+".csv")
    #
    # # trap coefficients of variation data
    # for key, val in Control.Trap_coeffs_of_vars_dict.items():
    #     val.to_csv("Control dataset data/Trap Coefficients of variation data/Trap " + str(key) + ".csv")
    #
    # # the intra_generational data
    # for ind in range(len(Control.A_intra_gen_bacteria)):
    #     Control.A_intra_gen_bacteria[ind].to_csv("Control dataset data/Intra-generationional data/A trace of intra-generational data with "+str(ind)+" separation.csv")
    #     Control.B_intra_gen_bacteria[ind].to_csv("Control dataset data/Intra-generationional data/B trace of intra-generational data with "+str(ind)+" separation.csv")
    #
    # # trace statistics
    # for key, val in Control.A_stats_dict.items():
    #     val.to_csv("Control dataset data/Trace Statistics data/A "+str(key)+".csv")
    #
    # for key, val in Control.B_stats_dict.items():
    #     val.to_csv("Control dataset data/Trace Statistics data/B "+str(key)+".csv")
    #
    # # trap statistics
    # for key, val in Control.trap_stats_dict.items():
    #     val.to_csv("Control dataset data/Trap Statistics data/Trap " + str(key) + ".csv")




    # os.mkdir("Sister dataset data")
    # os.mkdir("Sister dataset data/Protein data")
    # os.mkdir("Sister dataset data/Generation data")
    # os.mkdir("Sister dataset data/Intra-generationional data")
    # os.mkdir("Sister dataset data/Trace Coefficients of variation data")
    # os.mkdir("Sister dataset data/Trap Coefficients of variation data")
    # os.mkdir("Sister dataset data/Trace Statistics data")
    # os.mkdir("Sister dataset data/Trap Statistics data")
    # 
    # # Now for Sister!
    # # protein
    # for key, val in Sister.A_raw_protein_data_dict.items():
    #     val.to_csv("Sister dataset data/Protein data/A trace " + str(key) + ".csv")
    #     
    # for key, val in Sister.B_raw_protein_data_dict.items():
    #     val.to_csv("Sister dataset data/Protein data/B trace " + str(key) + ".csv")
    # 
    # # generation data
    # for key, val in Sister.A_dict.items():
    #     val.to_csv("Sister dataset data/Generation data/A trace " + str(key) + ".csv")
    #     
    # for key, val in Sister.B_dict.items():
    #     val.to_csv("Sister dataset data/Generation data/B trace " + str(key) + ".csv")
    # 
    # # trace coefficients of variation data
    # for key, val in Sister.A_coeffs_of_vars_dict.items():
    #     val.to_csv("Sister dataset data/Trace Coefficients of variation data/A trace " + str(key) + ".csv")
    # 
    # for key, val in Sister.B_coeffs_of_vars_dict.items():
    #     val.to_csv("Sister dataset data/Trace Coefficients of variation data/B trace " + str(key) + ".csv")
    # 
    # # trap coefficients of variation data
    # for key, val in Sister.Trap_coeffs_of_vars_dict.items():
    #     val.to_csv("Sister dataset data/Trap Coefficients of variation data/Trap " + str(key) + ".csv")
    # 
    # # the intra_generational data
    # for ind in range(len(Sister.A_intra_gen_bacteria)):
    #     Sister.A_intra_gen_bacteria[ind].to_csv(
    #         "Sister dataset data/Intra-generationional data/A trace of intra-generational data with " + str(ind) + " separation.csv")
    #     Sister.B_intra_gen_bacteria[ind].to_csv(
    #         "Sister dataset data/Intra-generationional data/B trace of intra-generational data with " + str(ind) + " separation.csv")
    # 
    # # trace statistics
    # for key, val in Sister.A_stats_dict.items():
    #     val.to_csv("Sister dataset data/Trace Statistics data/A " + str(key) + ".csv")
    # 
    # for key, val in Sister.B_stats_dict.items():
    #     val.to_csv("Sister dataset data/Trace Statistics data/B " + str(key) + ".csv")
    # 
    # # trap statistics
    # for key, val in Sister.trap_stats_dict.items():
    #     val.to_csv("Sister dataset data/Trap Statistics data/Trap " + str(key) + ".csv")
        
        
        
        
        

    # os.mkdir("Nonsister dataset data")
    # os.mkdir("Nonsister dataset data/Protein data")
    # os.mkdir("Nonsister dataset data/Generation data")
    # os.mkdir("Nonsister dataset data/Intra-generationional data")
    # os.mkdir("Nonsister dataset data/Trace Coefficients of variation data")
    # os.mkdir("Nonsister dataset data/Trap Coefficients of variation data")
    # os.mkdir("Nonsister dataset data/Trace Statistics data")
    # os.mkdir("Nonsister dataset data/Trap Statistics data")
    #
    # # Now for Nonsister!
    # # protein
    # for key, val in Nonsister.A_raw_protein_data_dict.items():
    #     val.to_csv("Nonsister dataset data/Protein data/A trace " + str(key) + ".csv")
    #
    # for key, val in Nonsister.B_raw_protein_data_dict.items():
    #     val.to_csv("Nonsister dataset data/Protein data/B trace " + str(key) + ".csv")
    #
    # # generation data
    # for key, val in Nonsister.A_dict.items():
    #     val.to_csv("Nonsister dataset data/Generation data/A trace " + str(key) + ".csv")
    #
    # for key, val in Nonsister.B_dict.items():
    #     val.to_csv("Nonsister dataset data/Generation data/B trace " + str(key) + ".csv")
    #
    # # trace coefficients of variation data
    # for key, val in Nonsister.A_coeffs_of_vars_dict.items():
    #     val.to_csv("Nonsister dataset data/Trace Coefficients of variation data/A trace " + str(key) + ".csv")
    #
    # for key, val in Nonsister.B_coeffs_of_vars_dict.items():
    #     val.to_csv("Nonsister dataset data/Trace Coefficients of variation data/B trace " + str(key) + ".csv")
    #
    # # trap coefficients of variation data
    # for key, val in Nonsister.Trap_coeffs_of_vars_dict.items():
    #     val.to_csv("Nonsister dataset data/Trap Coefficients of variation data/Trap " + str(key) + ".csv")
    #
    # # the intra_generational data
    # for ind in range(len(Nonsister.A_intra_gen_bacteria)):
    #     Nonsister.A_intra_gen_bacteria[ind].to_csv(
    #         "Nonsister dataset data/Intra-generationional data/A trace of intra-generational data with " + str(ind) + " separation.csv")
    #     Nonsister.B_intra_gen_bacteria[ind].to_csv(
    #         "Nonsister dataset data/Intra-generationional data/B trace of intra-generational data with " + str(ind) + " separation.csv")
    #
    # # trace statistics
    # for key, val in Nonsister.A_stats_dict.items():
    #     val.to_csv("Nonsister dataset data/Trace Statistics data/A " + str(key) + ".csv")
    #
    # for key, val in Nonsister.B_stats_dict.items():
    #     val.to_csv("Nonsister dataset data/Trace Statistics data/B " + str(key) + ".csv")
    #
    # # trap statistics
    # for key, val in Nonsister.trap_stats_dict.items():
    #     val.to_csv("Nonsister dataset data/Trap Statistics data/Trap " + str(key) + ".csv")


if __name__ == '__main__':
    main()
