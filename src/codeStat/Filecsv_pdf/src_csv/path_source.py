import sys

list_path = [
    'D:/doc/2026/FINAL_CODE/src',
    'D:/doc/2026/FINAL_CODE/src/codeStat/modelling',
    'D:/doc/2026/FINAL_CODE/src/codeStat/Filecsv_pdf',
    'D:/doc/2026/FINAL_CODE/src/codeStat/modelimports',
    'D:/doc/2026/FINAL_CODE/src/codeStat/modelling/languageModelling',
    'D:/doc/2026/FINAL_CODE/src/codeStat/modelling/voiceModelling',
    'D:/doc/2026/FINAL_CODE/src/codeStat/modelling/voiceModelling/srcVoiceModelling',
    'D:/doc/2026/FINAL_CODE/src/codeStat/modelling/voiceModelling/srcVoiceModelling/hybrid',
    'D:/doc/2026/FINAL_CODE/src/codeStat/modelling/voiceModelling/srcVoiceModelling/TDNN',
    'D:/doc/2026/FINAL_CODE/src/codeStat/pastModelling',
    'D:/doc/2026/FINAL_CODE/src/codeStat/preModelling',
    'D:/doc/2026/FINAL_CODE/src/connection'
]

for path in list_path:
    if path not in sys.path:
        sys.path.append(path)

print(sys.path)