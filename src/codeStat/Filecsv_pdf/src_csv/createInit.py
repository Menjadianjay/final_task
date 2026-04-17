import os

base = 'D:/doc/2026/FINAL_CODE/src/codeStat'
base_connect='D:/doc/2026/FINAL_CODE/src/dbMongo'
for dirpath, dirnames, filenames in os.walk(base_connect):
    init_file = os.path.join(dirpath, '__init__.py')
    if not os.path.exists(init_file):
        open(init_file, 'w').close()
        print(f"Created: {init_file}")