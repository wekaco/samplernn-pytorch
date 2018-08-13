
import os
import glob
import argparse
from timeit import default_timer as timer
from scipy.io import wavfile
import numpy as np
import utils
from collections import defaultdict
import librosa

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
    str = 'All files have the same size'
    same_size = True
    if len(files_size_dict.keys()) >= 2:
        same_size = False
        str = ''
        for k, v in files_size_dict.items():
            str += '{} files of size {} found.\n'.format(v, k)
    print(str)
    return same_size


def pad_audio_files(files, args, padding=48000):
    print("Pad audio files to total size of {}".format(padding))
    sr_dir = '{}_pad/'.format(args.folder)
    if not os.path.exists(sr_dir):
        os.mkdir(sr_dir)

    maxv = np.iinfo(np.int16).max
    for f in files:
        filename = f.split('/')[-1]
        print(filename)
        # fs, data = wavfile.read(f)
        data, fs = librosa.load(f)
        f_length = np.shape(data)[0]
        if f_length < padding:
            padded_data = np.zeros([padding])
            padded_data[0:f_length] = data
            print("Data: {}, padded_data: {}".format(np.shape(data), np.shape(padded_data)))
        else:  # trim data
            padded_data = data[0:padding]
        # wavfile.write('{}{}'.format(sr_dir, filename), fs, padded_data)
        librosa.output.write_wav('{}{}'.format(sr_dir, filename), (padded_data*maxv).astype(np.int16), fs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("--folder", default="datasets/splices_audio_BMI_16000_c1_16bits_music_eq", help="Path to input audio file.")
    parser.add_argument("--folder", default="datasets/COGNIMUSE_eq_eq",
                        help="Path to input audio file.")
    #parser.add_argument("--folder", default="datasets/piano3",
    #                    help="Path to input audio file.")
    parser.add_argument("--audio_delimiter", default=".wav", help="Audio encoding.")
    parser.add_argument("--bits", default=8, help="Number of bits in audio.")
    # parser.add_argument("--check", action="store_const", const=True, default=False, help="Check audio files size.")

    args = parser.parse_args()

    init_time = timer()

    IFS = args.audio_delimiter  # delimiter
    print(IFS)
    print(args.folder)

    # Calculate max samples, gives notice! Check audio segmentation to see why audio is non existent
    files = glob.glob("{}/*{}".format(args.folder, IFS))
    print(files)
    all_same = check_audio_files_size(files)
    if not all_same:
        pad_audio_files(files, args, padding=48000)  # 3seconds * 16KHz

    '''
    # Change sample rate to 16000, 1 channel, B bits
    files = glob.glob(args.folder+"/*{}".format(IFS))
    for f in files:
        f_base = f.split(IFS)[0]
        filename = f_base.split('/')[-1]
        print(f_base)

        sr_dir = '{}_16000_c1_{}bits/'.format(args.folder, args.bits)
        if not os.path.exists(sr_dir):
            os.mkdir(sr_dir)
        os.system('sox {f} -c1 -b{bits} -r16000 {sr_dir}{filename}{IFS}'.
                  format(f=f, bits=args.bits, sr_dir=sr_dir, filename=filename, IFS=IFS))
    '''
    end_time = timer()
    print("Program took {} minutes".format((end_time-init_time)/60))  # COGNIMUSE: 47 minutes