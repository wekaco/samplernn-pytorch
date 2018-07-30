
import os
import glob
import argparse
from timeit import default_timer as timer
from scipy.io import wavfile
import numpy as np
import utils
from collections import defaultdict

"""
Transform wav files in folder to 16kHz, 8 bits, 1 channel audio files
"""


def check_audio_files_size(files):
    print("Check audio files size")
    files_size_dict = defaultdict(lambda: 0)
    for f in files:
        fs, data = wavfile.read(f)
        f_length = np.shape(data)[0]
        files_size_dict[f_length] += 1

    print(files_size_dict.items())
    str = 'All sizes are the same'
    same_size = True
    if len(files_size_dict.keys()) >= 2:
        str = ''
        for k, v in files_size_dict.items():
            str += '{} files of size {} found.\n'.format(v, k)
    print(str)
    return same_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="datasets/splices_audio_BMI", help="Path to input audio file.")
    parser.add_argument("--audio_delimiter", default=".wav", help="Audio encoding.")

    args = parser.parse_args()

    init_time = timer()

    IFS = args.audio_delimiter  # delimiter
    print(IFS)
    print(args.folder)

    # Calculate max samples, gives notice! Check audio segmentation to see why audio is non existent
    files = glob.glob("{}/*{}".format(args.folder, IFS))
    all_same = check_audio_files_size(files)

    # Change sample rate to 16000, 1 channel, 8 bits
    files = glob.glob(args.folder+"/*{}".format(IFS))
    for f in files:
        f_base = f.split(IFS)[0]
        filename = f_base.split('/')[-1]
        print(f_base)

        sr_dir = '{}_16000_c1_8bits/'.format(args.folder)
        if not os.path.exists(sr_dir):
            os.mkdir(sr_dir)
        os.system('sox {f} -c1 -b8 -r16000 {sr_dir}{filename}{IFS}'.
                  format(f=f, sr_dir=sr_dir, filename=filename, IFS=IFS))

    end_time = timer()
    print("Program took {} minutes".format((end_time-init_time)/60))  # COGNIMUSE: 47 minutes