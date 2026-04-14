import os
os.chdir(r"D:\doc\2026\FINAL_CODE\src")

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
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmm as GMM
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmm import *
#GMM_PIPELINE
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmmpipeline as GMM_pipeline
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.gmmpipeline import *
#Lexicon_Ngram
import codeStat.modelling.languageModelling.srcLanguageModelling.lexicon_ngram as lexicon_ngram
from codeStat.modelling.languageModelling.srcLanguageModelling.lexicon_ngram import *

# Test model 3
# Pre
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.pre.ekstraksiMFCC as ekstraksiPRE
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.pre.ekstraksiMFCC import *
# model
    # GMMMODEL
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.model.gmmModel as gmmModel
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.model.gmmModel import *
    # TRAINMODEL
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.model.trainGMM as trainGMM
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.model.trainGMM import *
# past
    # hyperparameter Tunning
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.hyperparams as hyperParams
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.hyperparams import *
    # Language NGram Model
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.ngram as ngram
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.ngram import *
    # Normalisasi Lexicon
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.normalisasiLexicon as normLexicon
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.normalisasiLexicon import *
    # Word Error Rate
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.WER as wer
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.past.WER import *
# GMM Import 
import codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.impgmm as impLib
from codeStat.modelling.voiceModelling.srcVoiceModelling.hybrid.src.GMM.test.prefix.impgmm import *