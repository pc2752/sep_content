import models_autovc as models
import tensorflow as tf
import argparse
import os, sys
import config_autovc as config
import utils
import numpy as np
import mir_eval

def load_model():
    if config.op_mode == "autovc":
        model = models.AutoVC()
    elif config.op_mode == "sep_content":
        model = models.SSSynth_Content()
    elif config.op_mode == "sep":
        model = models.SSSynth()
    return model

def eval_hdf5_file(file_name, speaker_alt = None):

    model = load_model()

    speaker_name = file_name.split('_')[1]
    speaker_index = config.singers.index(speaker_name)
    if not speaker_alt:
        model.test_file_hdf5(file_name, speaker_index, speaker_index)
    else:
        model.test_file_hdf5(file_name, speaker_index, speaker_alt)
 

def eval_wav_file(file_name, acap_file=None):
    model = load_model()
    model.test_file_wav(file_name, acap_file)

def eval_wav_folder(folder_name):
    model = load_model()
    model.test_folder_wav(folder_name)

def train(_):


    model = load_model()

    model.train()


if __name__ == '__main__':
    if len(sys.argv)<2 or sys.argv[1] == '-help' or sys.argv[1] == '--help' or sys.argv[1] == '--h' or sys.argv[1] == '-h':
        print("%s --help or -h or --h or -help to see this menu" % sys.argv[0])
        print("%s --train or -t or --t or -train to train the model" % sys.argv[0])
        print("%s -e or --e or -eval or --eval  <filename> to evaluate an hdf5 file" % sys.argv[0])
        print("%s -v or --v or -val or --val <filename> to evaluate a wav file" % sys.argv[0])

    else:
        if sys.argv[1] == '-train' or sys.argv[1] == '--train' or sys.argv[1] == '--t' or sys.argv[1] == '-t':
            print("Training")
            tf.app.run(main=train)

        elif sys.argv[1] == '-e' or sys.argv[1] == '--e' or sys.argv[1] == '--eval' or sys.argv[1] == '-eval':
            if len(sys.argv)<3:
                print("Please give a hdf5 file to evaluate")
            else:
                file_name = sys.argv[2]
                if not file_name.endswith('.hdf5'):
                    file_name = file_name+'.hdf5'
                if not file_name in [x for x in os.listdir(config.voice_dir)]:
                    print("Currently only supporting hdf5 files which are in the dataset, will be expanded later.")
                    print([x for x in os.listdir(config.voice_dir) if x.startswith('nus' )or x.startswith('casas') or x.startswith('med')])
                else:
                    if len(sys.argv)<4:
                        print("Synthesizing with same singer.")
                        eval_hdf5_file(file_name)
                    else:
                        speaker_alt = int(sys.argv[3])
                        assert speaker_alt <len(config.singers), "Please give a number between 0 and {}, the rest will be implemented soon".format(len(config.singers))
                        eval_hdf5_file(file_name, speaker_alt)
        elif sys.argv[1] == '-w' or sys.argv[1] == '--w':
            if len(sys.argv)<3:
                print("Please give a wav file to evaluate")
            else:
                file_name = sys.argv[2]
                if len(sys.argv) == 4:
                    acap_file = sys.argv[3]
                    eval_wav_file(file_name, acap_file)
                else:
                    eval_wav_file(file_name)
        elif sys.argv[1] == '-f' or sys.argv[1] == '--f':
            if len(sys.argv)<3:
                print("Please give a folder with wav files to evaluate")
            else:
                folder_name = sys.argv[2]
                eval_wav_folder(folder_name)

