import os
import re
import glob
import math
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


try:
    import librosa
    _LIBROSA = True
except ImportError:
    librosa = None
    _LIBROSA = False
    print("[WARNING] librosa not installed")

try:
    import scipy.io.wavfile as _wavfile
    _SCIPY = True
except ImportError:
    _wavfile = None
    _SCIPY = False
    print("[WARNING] scipy not installed")