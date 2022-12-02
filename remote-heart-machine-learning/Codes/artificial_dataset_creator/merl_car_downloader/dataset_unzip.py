# %%
from zipfile38 import ZipFile
import os
import numpy as np
# %%
filename = '/home/desafio01/Documents/Codes/bio_hmd/Dataset_MR_NIRP/Subject3_still_940.zip'
file_path = '/home/desafio01/Documents/Codes/bio_hmd/Dataset_MR_NIRP/MR-NIRP_Indoor/Subject3_still_940'

# Using zipfile38.ZipFile to extract ZIPs that can't be opened otherwise
with ZipFile(os.path.join(file_path, filename), 'r') as zipfile:
    zipfile.extractall(path=file_path)
print('Done')
# %%
# files_qtd_frames = {}

# Getting the amount of files at each ZIP
# for dirpath, dirnames, filenames in os.walk('MR-NIRP_Car'):
#     for filename in filenames:
#         with ZipFile(os.path.join(dirpath, filename), 'r') as zipfile:
#             print(f"{os.path.join(dirpath, filename)}: {len(zipfile.filelist)}")
#             file_sum = files_qtd_frames.get(filename, [])
#             file_sum.append(len(zipfile.filelist))
#             files_qtd_frames[filename] = file_sum
#         print('=' * 30)

# for file_type, qtds in files_qtd_frames.items():
#     print(f'Media de frames do {file_type}: {np.mean(qtds)}')
