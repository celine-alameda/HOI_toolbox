# Author: Pranav Mahajan, 2021

import numpy as np
import pandas as pd
from numpy import genfromtxt
import scipy.io
import time
import json
import sys
import os

from toolbox.utils import save_obj, load_obj
from toolbox.Oinfo import exhaustive_loop_zerolag
from toolbox.dOinfo import exhaustive_loop_lagged
from toolbox.local_o_info import exhaustive_local_o

configFilename = "config.json"
argCount = len(sys.argv)
if (argCount > 1):
    configFilename = sys.argv[1]

outputDirectory = "output"
if (not os.path.exists(outputDirectory)):
    os.makedirs(outputDirectory)

with open(configFilename, "r") as fd:
    config = json.load(fd)

if ("metric" in config):
    metric = config["metric"]
else:
    print("ERROR : no metric specified")
    exit(1)

if ("input" in config):
    file_name = config["input"]
else:
    print("Please provide input data location in config")
    sys.exit()

if ("input_type" in config):
    input_type = config["input_type"]
else:
    input_type = "tsv"

if input_type == "tsv":
    # df = pd.read_csv("data/timeseries.tsv.gz", compression='gzip', delimiter='\t')
    # df = df.loc[:, (df != 0.0).any(axis=0)]
    # df.to_csv('data/cleaned_timeseries.tsv', sep='\t',index=False)
    file_name = config["input"]
    ts = genfromtxt(file_name, delimiter='\t', )
    ts = ts[1:, :].T  # 101 variables, 152 timepoints
# print(ts.shape)
elif input_type == "mat":
    ts = scipy.io.loadmat(file_name)
    ts = np.array(ts['ts'])
    ts = ts.T
# print(ts.shape)
else:
    print("Unknown input type")
    sys.exit()

if metric == "Oinfo":
    t = time.time()
    Odict = exhaustive_loop_zerolag(ts, config)
    elapsed = time.time() - t
    print("Elapsed time is ", elapsed, " seconds.")
    save_name = config["input"].split(".")[0]
    print("Saving and trying to load again")
    save_obj(Odict, save_name)
    Odict_Oinfo = load_obj('Odict_Oinfo')
    print("Done.")
elif metric == "dOinfo":
    print("WARNING : CHECK CODE TO SEE IF INPUT FILE IS CORRECT")
    t = time.time()
    Odict = exhaustive_loop_lagged(ts, config)
    elapsed = time.time() - t
    print("Elapsed time is ", elapsed, " seconds.")
    save_obj(Odict, 'Odict_dOinfo')
    Odict_dOinfo = load_obj('Odict_dOinfo')
    print(Odict_dOinfo)

elif metric == "local_o":
    t = time.time()
    ts = pd.DataFrame(ts.transpose())
    local_o = exhaustive_local_o(ts)
    elapsed = time.time() - t
    print("Elapsed time is ", elapsed, " seconds.")
    save_name = config["input"].split(".")[0]+"_local"
    print("Saving and trying to load again")
    save_obj(local_o, save_name)
    local_o = load_obj(save_name)
    print(local_o)
    print("Done.")
else :
    print("ERROR : Unknown metric")
    exit(1)