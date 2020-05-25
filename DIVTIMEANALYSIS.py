



#CHECK THE DIVISION TIMES!
mydirA = 'check_division_times_length_non_sisters/A/'
mydirB = 'check_division_times_length_non_sisters/B/'

for dataid in range(len(A_dict_non_sis)):
    CheckDivTimes(mydir=mydirA, traj='A', Data=Nonsisters, dataid=dataid, dictionary=A_dict_non_sis)
for dataid in range(len(B_dict_non_sis)):
    CheckDivTimes(mydir=mydirB, traj='B', Data=Nonsisters, dataid=dataid, dictionary=B_dict_non_sis)


