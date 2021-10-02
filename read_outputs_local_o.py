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

data = pd.read_table(config["input"])
variable_names = list(data.columns)
save_name = config["input"].split(".")[0] + "_local"
local_o_info = load_obj(save_name)

print("Local o info readout, object {}".format(save_name))
print(local_o_info)

data["Local o"] = local_o_info
output_file_name = save_name + ".xlsx"
data.to_excel(output_file_name)
