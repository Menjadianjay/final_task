import sys
sys.path.insert(0, r"D:\doc\2026\FINAL_CODE\src") 
import os
import re
import glob
import math
import time
import soundfile as sf
import matplotlib
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



base_dir = 'D:/doc/2026/FINAL_CODE/'
all_data_root = f'{base_dir}data/Data'
sampled_root = f'{base_dir}data/sampled'
sampled_small_root = f'{base_dir}data/sampled_small'

#convertFile
import codeStat.Filecsv_pdf.src_convertingFile.converting_file as cnv
from codeStat.Filecsv_pdf.src_convertingFile.converting_file import *
#Sampling_ Quantization
import codeStat.preModelling.sampling_Quantization.src_Sampling_Quantization.Sampling_Quantization as Sampling_Quantization
from codeStat.preModelling.sampling_Quantization.src_Sampling_Quantization.Sampling_Quantization import *
#Voice Activity Detection
import codeStat.preModelling.voiceActivityDetection.src_VoiceActivityDetection.voiceActivityDetection as VAD
from codeStat.preModelling.voiceActivityDetection.src_VoiceActivityDetection.voiceActivityDetection import *
#Feature Extraction
import codeStat.preModelling.featureExtraction.src_featureExtraction.feature_extraction as feature_Extraction
from codeStat.preModelling.featureExtraction.src_featureExtraction.feature_extraction import *
#Feature Extraction2
import codeStat.preModelling.featureExtraction.src_featureExtraction.feature_extraction2 as feature_Extraction2
from codeStat.preModelling.featureExtraction.src_featureExtraction.feature_extraction2 import *
#GMM
import codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.gmm as GMM
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.gmm import *
#GMM_PIPELINE
import codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.gmmpipeline as GMM_pipeline
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.gmmpipeline import *
#Lexicon_Ngram
import codeStat.modelling.languageModelling.srcLanguageModelling.lexicon_ngram as lexicon_ngram
from codeStat.modelling.languageModelling.srcLanguageModelling.lexicon_ngram import *

# Test model 3
# Pre
import codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.pre.ekstraksiMFCC as ekstraksiPRE
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.pre.ekstraksiMFCC import *
# model
    # GMMMODEL
import codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.model.gmmModel as gmmModel
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.model.gmmModel import *
    # TRAINMODEL
import codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.model.trainGMM as trainGMM
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.model.trainGMM import *
# post
    # hyperparameter Tunning
import codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.post.hyperparams as hyperParams
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.post.hyperparams import *
    # Language NGram Model
import codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.post.ngram as ngram
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.post.ngram import *
    # Normalisasi Lexicon
import codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.post.normalisasiLexicon as normLexicon
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.post.normalisasiLexicon import *
    # Word Error Rate
import codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.post.WER as wer
from codeStat.modelling.voiceModelling.srcVoiceModelling.plan2.GMM.GMM.test.prefix.post.WER import *

   
__all__ = [
    "os", "re", "glob", "math", "time",
    "np", "pd","sf","matplotlib",
    "defaultdict",
    "Dict", "List", "Tuple", "Optional",
    "librosa","_LIBROSA","_SCIPY","_wavfile",

    "cnv",                  # converting_file
    "Sampling_Quantization",# sampling & quantization
    "VAD",                  # voice activity detection
    "feature_Extraction",   # feature extraction
    "feature_Extraction2",  # feature extraction 2
    "GMM",                  # gmm
    "GMM_pipeline",         # gmmpipeline
    "lexicon_ngram",        # lexicon & ngram

    "ekstraksiPRE",         # ekstraksiMFCC
    "gmmModel",             # gmmModel
    "trainGMM",             # trainGMM
    "hyperParams",          # hyperparameter tuning
    "ngram",                # language ngram model
    "normLexicon",          # normalisasi lexicon
    "wer",                  # word error rate
]