import os
os.chdir(r"D:\doc\2026\FINAL_CODE\src")

base_dir = 'D:/doc/2026/FINAL_CODE/'
all_data_root = f'{base_dir}data/Data'
sampled_root = f'{base_dir}data/sampled'
sampled_small_root = f'{base_dir}data/sampled_small'

#convertFile
import Filecsv_pdf.src_convertingFile.converting_file as cnv
from Filecsv_pdf.src_convertingFile.converting_file import *
#Sampling_ Quantization
import preModelling.sampling_Quantization.src_Sampling_Quantization.Sampling_Quantization as Sampling_Quantization
from preModelling.sampling_Quantization.src_Sampling_Quantization.Sampling_Quantization import *
#Voice Activity Detection
import preModelling.voiceActivityDetection.src_VoiceActivityDetection.voiceActivityDetection as VAD
from preModelling.voiceActivityDetection.src_VoiceActivityDetection.voiceActivityDetection import *
#Feature Extraction
import preModelling.featureExtraction.src_featureExtraction.feature_extraction as feature_Extraction
from preModelling.featureExtraction.src_featureExtraction.feature_extraction import *
#Feature Extraction2
import preModelling.featureExtraction.src_featureExtraction.feature_extraction2 as feature_Extraction2
from preModelling.featureExtraction.src_featureExtraction.feature_extraction2 import *
#GMM
import hybrid.src.GMM.test.gmm as GMM
from hybrid.src.GMM.test.gmm import *
#GMM_PIPELINE
import modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmmpipeline as GMM_pipeline
from modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmmpipeline import *
#Lexicon_Ngram
import modelling.languageModelling.srcLanguageModelling.lexicon_ngram as lexicon_ngram
from modelling.languageModelling.srcLanguageModelling.lexicon_ngram import *