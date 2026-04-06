import os
os.chdir(r"D:\doc\2026\FINAL_CODE\src")
print(os.getcwd())

all_data_root = 'D:/doc/2026/FINAL_CODE/data/Data'
sampled_root = 'D:/doc/2026/FINAL_CODE/data/sampled'
sampled_small_root = 'D:/doc/2026/FINAL_CODE/data/sampled_small'

#convertFile
import Filecsv_pdf.src_convertingFile.converting_file as cnv
from Filecsv_pdf.src_convertingFile.converting_file import *
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
