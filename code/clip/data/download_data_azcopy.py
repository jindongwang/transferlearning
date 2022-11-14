# download datasets from azure blob storage, which is faster than downloading from the website

import os
import argparse

DATA_DIR_AZURE = {
    'office-home': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/OfficeHome/',
    'office31': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/office31/',
    'domainnet': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/domainnet/',
    'visda': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/visda/',
    'vlcs': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/VLCS/VLCS/',
    'pacs': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/PACS/PACS/kfold/',
    'terrainc': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/terra_incognita/terra_incognita/',
    'wilds-camelyon': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/wilds/wilds/camelyon17_v1.0/',
    'wilds-fmow': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/wilds/wilds/fmow_v1.1/',
    'wilds-iwildcam': 'https://transferlearningdrive.blob.core.windows.net/teamdrive/dataset/wilds/wilds/iwildcam_v2.0/'
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=DATA_DIR_AZURE.keys(), default='office31')
    parser.add_argument('--to', type=str, default='/data/jindwang/', help='path to save the dataset')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(os.getcwd() + '/azcopy'):
        azcopy_path = 'https://wjdcloud.blob.core.windows.net/dataset/azcopy'
        print('>>>>Downloading azcopy...')
        os.system('wget {}'.format(azcopy_path))
        os.system('chmod +x azcopy')

    print('>>>>Downloading datasets...')
    src_path = DATA_DIR_AZURE[args.dataset]
    os.system(f'./azcopy copy "{src_path}" "{args.to}" --recursive')
    print('>>>>Done!')

