# %%
import subprocess
import pandas as pd
import os
import sys

def treating_data(path, filter_subject=None, to_download=None):
    """
    Function that treat and filter the Dataframe:
        - Filter by filename
        - Filter by subject
        - Get path from raw text
        - Remove unnecessary entries
        - Sort by subject number
        - pre-process the link
    """
    
    df = pd.read_csv(path, sep=',', header=0)
    
    # - Filter by filename
    to_download = to_download if to_download else list(df.Filename.unique())
    to_download = to_download if isinstance(to_download, list) else [to_download]

    df = df[df.Filename.isin(to_download)]

    # - Get path from raw text
    df['Folder Path'] = df['Folder Path'].apply(lambda x: str(x).replace(' > ', os.path.sep)
                                                                .replace(' >', os.path.sep)
                                                                .replace('Shared with me' + os.path.sep, '')
                                                                .replace(' ', '_'))

    # - Filter by subject
    if filter_subject is not None and isinstance(filter_subject, int):
        df['Subject number'] = df['Folder Path'].apply(lambda x: int(''.join(filter(str.isdigit, str(x).split(os.path.sep)[2]))))
        
        df = df[df['Subject number'] == filter_subject]

        # - Sort by subject number
        df = df.sort_values('Subject number', ignore_index=True)

    # - Remove unnecessary entries
    df = df[df.Filetype != 'Folder']

    # - preprocess the link
    df['Direct Link'] = df['Direct Link'].apply(lambda x: x.replace('uc?id=', 'file/d/')
                                                            .split('&')[0] + '/view?usp=sharing')

    return df

def execute(cmd):
    try:
        result = subprocess.run(
            cmd.split(), 
            stdout=sys.stdout, 
            )
    except Exception as e:
        print(e)

def download_from_dataframe(df, skip=True):
    for index, row in df.iterrows():
        print('=' * 30)
        print(f'Creating folders: {row["Folder Path"]}')
        os.makedirs(row['Folder Path'], exist_ok=True)
        if os.path.isfile(os.path.join(row['Folder Path'], row['File name'])) and skip:
            print(f'File {row["File name"]} already exists')
        else:
            print(f'Downloading file: {row["File name"]}')
            # Need an executable from Goodls to work
            cmd = f'.{os.path.sep}goodls_linux_amd64 -u {row["Direct Link"]} -e {row["File Extension"]} -m {row["Filetype"]} -d {row["Folder Path"]}'
            print(cmd)
            execute(cmd)
        # break

if __name__ == '__main__':
    # CSV file containing paths of folders and drive files that you want to download
    # path = 'mrnirp_folder_structure.csv'
    path = './ubfc_structure.csv'
    # Possible value(s) to the `to_download` list: ['NIR.zip', 'RGB.zip', 'PulseOX.zip', 'PulseOx.zip']
    df = treating_data(path, filter_subject=None, to_download=None)
    download_from_dataframe(df, skip=True)
# %%
