# Author: Pranav Mahajan, 2021

import json
import sys
import os

from toolbox.hoi_toolbox import HOIToolbox

configFilename = "config.json"
argCount = len(sys.argv)
if (argCount > 1):
    configFilename = sys.argv[1]

outputDirectory = "output"
if (not os.path.exists(outputDirectory)):
    os.makedirs(outputDirectory)

with open(configFilename, "r") as fd:
    config = json.load(fd)

tb = HOIToolbox(config)
tb.run()

# todo add read_outputs here
