
import os
import glob
import argparse
from timeit import default_timer as timer
from scipy.io import wavfile
import numpy as np
import utils

"""
Transform wav files in folder to 16kHz, 8 bits, 1 channel audio files
"""


def check_audio_files_size():
    print("Check")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="datasets/splices_audio_BMI", help="Path to input audio file.")
    parser.add_argument("--audio_delimiter", default=".wav", help="Audio encoding.")

    args = parser.parse_args()

    init_time = timer()

    IFS = args.audio_delimiter  # delimiter
    print(IFS)
    print(args.folder)

    # TODO: Calculate max samples, delete others? Check audio segmentation to see why audio is non existent
    files = glob.glob("{}/*{}".format(args.folder, IFS))
    all_same = check_audio_files_size()
    '''
    audio_length = 132301
    files = glob.glob("{}/*{}".format(args.folder, IFS))
    for f in files:
        print(f)
        fs, data = wavfile.read(f)
        f_length = np.shape(data)
        if 132301 != f_length[0]:
            print("File: {}, length: {}".format(f, f_length))
    '''

    '''
    # Make sure samples are the same size
    from pydub import AudioSegment
    num_chunks = float("inf")
    files = glob.glob("{}/*{}".format(args.folder, IFS))
    for f in files:
        sound = AudioSegment.from_file(f, 'wav')
        fs = sound.frame_rate
        sound_chunks = len(sound)
        if num_chunks > sound_chunks:
            num_chunks = sound_chunks
            print("File: {}, length: {}, {}".format(f, sound_chunks, sound.frame_count()))

    folder = args.folder + '_eq_segs'
    utils.ensure_dir(folder)
    for f in files:
        sound = AudioSegment.from_file(f)
        sound.frame_rate
        new_sound = sound[1:num_chunks]
        f_base = f.split(IFS)[0]
        filename = '{folder}/{filename}{IFS}'.format(folder=folder, filename=f_base.split('/')[-1], IFS=IFS)
        new_sound.export(filename, format="wav")

    '''
    # Make sure all samples have the same size
    #folder = args.folder + '_eq_segs'
    #files = glob.glob(args.folder + "/*{}".format(IFS))
    #for f in files:
    #    fs, data = wavfile.read(f)
    #    f_base = f.split(IFS)[0]
    #    filename = '{folder}/{filename}{IFS}'.format(folder=folder, filename=f_base.split('/')[-1], IFS=IFS)
    #    wavfile.write(filename, fs, data[1:audio_length])

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