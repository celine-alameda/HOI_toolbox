# Author: Pranav Mahajan, 2021
# Author : Céline Alameda, 2021

import os
import sys
import json
import pandas as pd
from toolbox.utils import load_obj

from toolbox.hoi_toolbox import HOIToolbox

# guard for multiprocessing
if __name__ == "__main__":

    configFilename = "config.json"
    argCount = len(sys.argv)
    if (argCount > 1):
        configFilename = sys.argv[1]

    outputDirectory = "output"
    if (not os.path.exists(outputDirectory)):
        os.makedirs(outputDirectory)

    with open(configFilename, "r") as fd:
        config = json.load(fd)

    inputFiles = []
    pre_processings = ["3_u", "3_f"]
    subjects = list(range(1, 6)) + list(range(7, 16))
    conditions_tdcs = ["real", "sham"]
    conditions_time = ["pre", "post"]
    for pre_processing in pre_processings:
        for subject in subjects:
            for condition_tdcs in conditions_tdcs:
                for condition_time in conditions_time:
                    fileName = pre_processing + "_" + str(subject) + "_" + condition_tdcs + "_" + condition_time + ".tsv"
                    inputFiles.append(fileName)
    for inputFile in inputFiles:
        print("local O computation for file {}".format(inputFile))
        config["input"] = inputFile
        tb = HOIToolbox(config)
        tb.run()
        print("Local o computation done. Assembling outputs...")
        # read outputs
        data = pd.read_table("data/" + config["input"])
        variable_names = list(data.columns)
        save_name = "local_o_" + config["input"].split(".")[0]
        local_o_info = load_obj(save_name)

        print("Local o info readout, object {}".format(save_name))

        data["local_o"] = local_o_info
        data.to_csv(save_name.split('.')[0] + ".tsv", sep='\t', index=False)
