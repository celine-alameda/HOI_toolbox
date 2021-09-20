# This python file has sample codes as to how to read the outputs generated for different configs.
# The output dictionary is slightly different whether you are using higher_order = False or True

# necessary imports
import numpy as np
import itertools

from toolbox.utils import load_obj, CombinationsManager

## NOTE: If you face any issues with loading pickled outputs then use pickle5 instead of pickle in utils.py


## Case 1: argument higher_order = False.
# Pregenerating all possible combinations, not using the combinatorial numbering system.
# To follow along,
# Run the toy_example.py on fmri timeseries of just first 10 variables for the following config
# "higher_order": false,
# "estimator": "gcmi",
# "modelorder":3,
# "maxsize":4,
# "n_best":10,
# "nboot":100

# Run for both Oinfo and dOinfo and you will have generated outputs Odict_Oinfo.pkl and Odict_dOinfo.pkl
# Load the dicts (equivalent of structs from MATLAB)
Odict_Oinfo = load_obj('Odict_Oinfo')

print("Oinfo readout example, higher_order = False")
# How to read the Oinfo output?
# Say you want to know the outputs for multiplet size 3
# then you would print the following

print(Odict_Oinfo[4])

n_variables = 14
nplet_size = 4
nplets_iter = itertools.combinations(range(1, n_variables + 1), nplet_size)
nplets = []
for nplet in nplets_iter:
    nplets.append(nplet)
C = np.array(nplets)


print("highest 4 syn : ")
for i in [125, 156, 647, 814,  71, 984, 942, 147, 262, 482]:
    print(C[i])
print("highest 4 red :")
for i in [862, 918, 989, 438, 493, 913, 778, 259, 995, 694]:
    print(C[i])