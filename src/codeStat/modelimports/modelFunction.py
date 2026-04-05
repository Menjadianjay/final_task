import os
os.chdir(r"D:\doc\2026\FINAL_CODE\src")
print(os.getcwd())

sampled_root = '../data/sampled'
sampled_small_root = '../data/sampled_small'

#Sampling_ Quantization
import codeStat.preModelling.sampling_Quantization.src_Sampling_Quantization.Sampling_Quantization as Sampling_Quantization
from codeStat.preModelling.sampling_Quantization.src_Sampling_Quantization.Sampling_Quantization import *
#Voice Activity Detection
import codeStat.preModelling.voiceActivityDetection.src_VoiceActivityDetection.voiceActivityDetection as VAD
from codeStat.preModelling.voiceActivityDetection.src_VoiceActivityDetection.voiceActivityDetection import *
#Feature Extraction
import codeStat.preModelling.featureExtraction.src_featureExtraction.feature_extraction as feature_Extraction
from codeStat.preModelling.featureExtraction.src_featureExtraction.feature_extraction import *
#GMM
import codeStat.modelling.voiceModelling.srcVoiceModelling.GMM_TDNN.src.GMM.test.gmm_func as GMM
from codeStat.modelling.voiceModelling.srcVoiceModelling.GMM_TDNN.src.GMM.test.gmm_func import *
