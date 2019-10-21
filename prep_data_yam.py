# from __future__ import division
import os,re
import collections
import soundfile as sf
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt
import sys
import h5py
import csv
import pyworld as pw

import librosa





import config
import utils



def main():

    unprocessable = []
    all_singers = []

    sub_dirs = next(os.walk(config.wav_dir_yam))[1]

    count_sub = 0

    freq_grid = librosa.cqt_frequencies(config.cqt_bins, config.fmin, config.bins_per_octave)

    f_bins = utils.grid_to_bins(freq_grid, 0.0, freq_grid[-1])

    n_freqs = len(freq_grid)

    for sub_dir in sub_dirs:
        sub_directory = os.path.join(config.wav_dir_yam, sub_dir)
        singers = next(os.walk(sub_directory))[1]
        count_singers = 0

        for singer in singers:
            count_singers+=1
            all_singers.append(singer)
            singer_directory = os.path.join(sub_directory, singer)
            songs = next(os.walk(singer_directory))[1]
            count = 0
            for song in songs:
                

                song_dir = os.path.join(singer_directory, song)
                song_files = os.listdir(song_dir)
                back_wav_files = [x for x in song_files if x.endswith('Back.wav')]
                for back_wav_file in back_wav_files:
                    song_name = back_wav_file.split('_Back.wav')[0]
                    if song_name+'_Vo.wav' in song_files:
                        voc_wav_file = song_name+'_Vo.wav'
                        process = True
                    elif song_name+'_VoU87.wav' in song_files:
                        voc_wav_file = song_name+'_VoU87.wav'
                        process = True
                    else:
                        process = False


                    if process:
                        voc_segments, back_segments = utils.slice_data(song_dir, voc_wav_file, back_wav_file)
                        count_segments = 0

                        for i in range(len(voc_segments)):

                            name = 'yam_{}_{}.hdf5'.format(song_name, i)
                            count_segments+=1
                            utils.progress(count_segments,len(voc_segments), "{} songs of {}, {} singers of {} and {} sets of {} done".format(count, len(songs), count_singers, len(singers), count_sub, len(sub_dirs)))


                            if name not in os.listdir(config.voice_dir):

                                try:

                               
                                    voc_stft, voc_mel, world_feats, back_stft, atb = utils.process_data(voc_segments[i], back_segments[i], f_bins, n_freqs)
                             

                                    with h5py.File(config.voice_dir+name, mode='w') as hdf5_file:
                                    
                                        hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)

                                        hdf5_file.create_dataset("voc_stft_real", voc_stft.shape, np.float32)

                                        hdf5_file.create_dataset("voc_stft_image", voc_stft.shape, np.float32)

                                        hdf5_file.create_dataset("voc_mel", voc_mel.shape, np.float32)

                                        hdf5_file.create_dataset("world_feats", world_feats.shape, np.float32)

                                        hdf5_file.create_dataset("back_stft", back_stft.shape, np.float32)

                                        hdf5_file.create_dataset("atb", atb.shape, np.float32)


                                        hdf5_file["voc_stft"][:,:] = abs(voc_stft)

                                        hdf5_file["voc_stft_real"][:,:] = np.real(voc_stft)

                                        hdf5_file["voc_stft_image"][:,:] = np.imag(voc_stft)

                                        hdf5_file["voc_mel"][:,:] = voc_mel

                                        hdf5_file["world_feats"][:,:] = world_feats

                                        hdf5_file["back_stft"][:,:] = abs(back_stft)

                                        hdf5_file["atb"][:,:] = atb
                                # import pdb;pdb.set_trace()
                                except:
                                    unprocessable.append(name)
                                    

                    count+=1


        count_sub+=1
    print(unprocessable)

    import pdb;pdb.set_trace()
   


if __name__ == '__main__':
    main()