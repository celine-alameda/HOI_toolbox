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
o_info_dict = load_obj('Odict_Oinfo')

print("Oinfo readout example, higher_order = False")
# How to read the Oinfo output?
# Say you want to know the outputs for multiplet size 3
# then you would print the following


n_variables = 5
nplet_size = 3

print(o_info_dict[nplet_size])
nplets_iter = itertools.combinations(range(1, n_variables + 1), nplet_size)
nplets = []
for nplet in nplets_iter:
    nplets.append(nplet)
C = np.array(nplets)

red_indices = o_info_dict[nplet_size]["index_red"]
red_values = o_info_dict[nplet_size]["sorted_red"]
print("Highest redundancies for nplet of size {} :".format(nplet_size))
for i in range(len(red_indices)):
    print("{} with value {}".format(C[red_indices[i]], red_values[i]))

syn_indices = o_info_dict[nplet_size]["index_syn"]
syn_values = o_info_dict[nplet_size]["sorted_syn"]
print("Highest synergies for nplet of size {} :".format(nplet_size))
for i in range(len(syn_indices)):
    print("{} with value {}".format(C[syn_indices[i]], syn_values[i]))