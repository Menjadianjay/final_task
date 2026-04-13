import os
os.chdir(r"D:\doc\2026\FINAL_CODE\src")

all_data_root = 'D:/doc/2026/FINAL_CODE/data/Data'
sampled_root = 'D:/doc/2026/FINAL_CODE/data/sampled'
sampled_small_root = 'D:/doc/2026/FINAL_CODE/data/sampled_small'

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
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmm as GMM
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmm import *
#GMM_PIPELINE
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmmpipeline as GMM_pipeline
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmmpipeline import *
#Lexicon_Ngram
import codeStat.modelling.languageModelling.srcLanguageModelling.lexicon_ngram as lexicon_ngram
from codeStat.modelling.languageModelling.srcLanguageModelling.lexicon_ngram import *