
# featureExtraction
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.pre.ekstraksiMFCC as ekstraksiMFCC
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.pre.ekstraksiMFCC import *
# GMMmodelTrain
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.model.trainGMM as trainGMM
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.model.trainGMM import *



def import_libraries():
    libs = {}
    
    optional_libs = {
        "_LIBROSA": "librosa",
        "_SCIPY": "scipy.io.wavfile",
    }

    for lib, module_path in optional_libs.items():
        try:
            module = __import__(module_path, fromlist=["*"])
            libs[lib] = module
        except ImportError:
            libs[lib] = None
            print(f"[WARNING] {lib} not installed")

    import os
    import re
    import glob
    import math
    import time
    import numpy as np
    import pandas as pd
    from collections import defaultdict
    from typing import Dict, List, Tuple, Optional

    libs.update({
        "os": os,
        "re": re,
        "glob": glob,
        "math": math,
        "time": time,
        "np": np,
        "pd": pd,
        "defaultdict": defaultdict,
        "Dict": Dict,
        "List": List,
        "Tuple": Tuple,
        "Optional": Optional,
    })

    return libs



