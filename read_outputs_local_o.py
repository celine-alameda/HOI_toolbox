# This python file has sample codes as to how to read the outputs generated for different configs.
# The output dictionary is slightly different whether you are using higher_order = False or True

# necessary imports
import sys
import json
import numpy as np
import itertools
import pandas as pd

from toolbox.utils import load_obj, CombinationsManager

configFilename = "config.json"
argCount = len(sys.argv)
if (argCount > 1):
    configFilename = sys.argv[1]
with open(configFilename, "r") as fd:
    config = json.load(fd)

data = pd.read_table("data/"+config["input"])
variable_names = list(data.columns)
save_name = "local_o_" + config["input"].split(".")[0]
local_o_info = load_obj(save_name)

print("Local o info readout, object {}".format(save_name))

data = pd.DataFrame(local_o_info).transpose()
data.to_csv(save_name.split('.')[0]+".tsv", sep='\t', index=False)

