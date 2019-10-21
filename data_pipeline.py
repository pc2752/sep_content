import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt
import collections
import config
import utils


def gen_train_val():
    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('yam')]




    train_list = voc_list[:int(len(voc_list)*0.9)]
    val_list = voc_list[int(len(voc_list)*0.9):]

    utils.list_to_file(val_list,'./val_files.txt')

    utils.list_to_file(train_list,'./train_files.txt')

def data_gen_mask(mode = 'Train'):



    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('yam')]


    train_list = voc_list[:int(len(voc_list)*config.split)]

    val_list = voc_list[int(len(voc_list)*config.split):]

#    import pdb;pdb.set_trace()


    if config.use_casas:

        stat_file = h5py.File('./stats.hdf5', mode='r')
    else:
        stat_file = h5py.File('./stats_yam.hdf5', mode='r')

    max_feat = stat_file["feats_maximus"][()]
    min_feat = stat_file["feats_minimus"][()]

    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":
        batches = config.batches_per_epoch_train
        file_list = train_list
    else:
        batches = config.batches_per_epoch_val
        file_list = val_list

    for k in range(batches):

        mixes = []
        vocals = []
        backs = []
        out_feats = []
        targets_speakers = []
        # start_time = time.time()

        for i in range(max_files_to_process):

            file_index = np.random.randint(0,len(file_list))

            tr_file = train_list[file_index]

            with h5py.File(config.voice_dir+tr_file, "r") as voc_file:
                voc_stft = voc_file["voc_stft"][()]
                back_stft = voc_file["back_stft"][()]
                mel = voc_file["world_feats"][()]


            speaker_name = tr_file.split('_')[1]
            speaker_index = config.singers.index(speaker_name)

            mel = (mel - min_feat)/(max_feat-min_feat)

            for j in range(config.samples_per_file):


                voc_idx = np.random.randint(0,len(voc_stft)-config.max_phr_len)

                gain_back = np.clip(np.random.rand(1), 0.0, 0.6)

                gain_voc = np.clip(np.random.rand(1), 0.5,1.0)

                voc = voc_stft[voc_idx:voc_idx+config.max_phr_len,:]*gain_voc

                back = back_stft[voc_idx:voc_idx+config.max_phr_len,:]*gain_back

                mix_stft = (voc + back)

                mixes.append(mix_stft)
                vocals.append(voc)
                backs.append(back)
                out_feats.append(mel[voc_idx:voc_idx+config.max_phr_len,:])
                targets_speakers.append(speaker_index)



        mixes = np.array(mixes)
        vocals = np.array(vocals)

        backs = np.array(backs)
        out_feats = np.array(out_feats)
        targets_speakers = np.array(targets_speakers)

        yield mixes, vocals, backs, out_feats, targets_speakers


def data_gen_chain(mode = 'Train'):



    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('yam')]


    train_list = voc_list[:int(len(voc_list)*config.split)]

    val_list = voc_list[int(len(voc_list)*config.split):]

#    import pdb;pdb.set_trace()


    if config.use_casas:

        stat_file = h5py.File('./stats.hdf5', mode='r')
    else:
        stat_file = h5py.File('./stats_yam.hdf5', mode='r')

    max_feat = stat_file["feats_maximus"][()]
    min_feat = stat_file["feats_minimus"][()]

    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":
        batches = config.batches_per_epoch_train
        file_list = train_list
    else:
        batches = config.batches_per_epoch_val
        file_list = val_list

    for k in range(batches):

        mixes = []
        vocals = []
        backs = []
        targets = []
        # start_time = time.time()

        for i in range(max_files_to_process):

            file_index = np.random.randint(0,len(file_list))

            tr_file = train_list[file_index]

            with h5py.File(config.voice_dir+tr_file, "r") as voc_file:


                voc_stft = voc_file["voc_stft"][()]

                back_stft = voc_file["back_stft"][()]

                feats = voc_file['world_feats'][()]


            for j in range(config.samples_per_file):


                voc_idx = np.random.randint(0,len(voc_stft)-config.max_phr_len)

                gain_back = np.clip(np.random.rand(1), 0.0, 0.6)

                gain_voc = np.clip(np.random.rand(1), 0.5,1.0)

                voc = voc_stft[voc_idx:voc_idx+config.max_phr_len,:]*gain_voc

                back = back_stft[voc_idx:voc_idx+config.max_phr_len,:]*gain_back

                mix_stft = (voc + back)

                mixes.append(mix_stft)
                vocals.append(voc)
                backs.append(back)
                targets.append(feats[voc_idx:voc_idx+config.max_phr_len,:])



        mixes = np.array(mixes)
        vocals = np.array(vocals)

        backs = np.array(backs)

        targets = (targets-min_feat)/(max_feat-min_feat)

        yield mixes, vocals, backs, targets


def data_gen(mode = 'Train'):



    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('yam')]


    train_list = voc_list[:int(len(voc_list)*config.split)]

    val_list = voc_list[int(len(voc_list)*config.split):]

#    import pdb;pdb.set_trace()


    if config.use_casas:

        stat_file = h5py.File('./stats.hdf5', mode='r')
    else:
        stat_file = h5py.File('./stats_yam.hdf5', mode='r')

    max_feat = stat_file["feats_maximus"][()]
    min_feat = stat_file["feats_minimus"][()]

    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":
        batches = config.batches_per_epoch_train
        file_list = train_list
    else:
        batches = config.batches_per_epoch_val
        file_list = val_list

    for k in range(batches):

        inputs = []
        targets = []
        if config.f0_mode == "discrete":
            atbs = []

        # start_time = time.time()

        for i in range(max_files_to_process):

            file_index = np.random.randint(0,len(file_list))

            tr_file = train_list[file_index]

            with h5py.File(config.voice_dir+tr_file, "r") as voc_file:

                feats = voc_file['world_feats'][()]

                voc_stft = voc_file["voc_stft"][()]

                back_stft = voc_file["back_stft"][()]

                if config.f0_mode == "discrete":
                    atb = voc_file["atb"][()]
                    atb = atb[:, 1:]
                    atb[:, 0:4] = 0
                    atb = np.clip(atb, 0.0, 1.0)

            for j in range(config.samples_per_file):


                voc_idx = np.random.randint(0,len(voc_stft)-config.max_phr_len)

                gain_back = np.clip(np.random.rand(1), 0.0, 0.6)

                gain_voc = np.clip(np.random.rand(1), 0.5,1.0)

                mix_stft = (voc_stft[voc_idx:voc_idx+config.max_phr_len,:]*gain_voc + back_stft[voc_idx:voc_idx+config.max_phr_len,:]*gain_back)*(gain_back+gain_voc)

                inputs.append(mix_stft)

                targets.append(feats[voc_idx:voc_idx+config.max_phr_len,:])
                if config.f0_mode == "discrete":
                    atbs.append(atb[voc_idx:voc_idx+config.max_phr_len,:])


        targets = np.array(targets)
        inputs = np.array(inputs)

        targets = (targets-min_feat)/(max_feat-min_feat)
        inputs = np.clip(inputs, 0.0, 1.0)

        if config.f0_mode == "discrete":
            yield inputs, targets, atbs
        else:
            yield  inputs, targets, None

def data_gen_stft(mode = 'Train', sec_mode = 0):

    if config.use_casas:

        stat_file = h5py.File('./stats.hdf5', mode='r')
    else:
        stat_file = h5py.File('./stats_yam.hdf5', mode='r')

    max_feat = stat_file["feats_maximus"][()]
    min_feat = stat_file["feats_minimus"][()]


    stat_file.close()  




    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('yam')]




    train_list = voc_list[:int(len(voc_list)*0.9)]
    val_list = voc_list[int(len(voc_list)*0.9):]



    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":
        num_batches = config.batches_per_epoch_train
        file_list = train_list

    else: 
        num_batches = config.batches_per_epoch_val
        file_list = val_list

    for k in range(num_batches):
        feats_targs = []

        targets_speakers = []

        stft_targs = []

        for i in range(max_files_to_process):


            voc_index = np.random.randint(0,len(file_list))
            voc_to_open = file_list[voc_index]


            with h5py.File(config.voice_dir+voc_to_open, "r") as hdf5_file:
                mel = hdf5_file["world_feats"][()]
                if config.mix_aug:
                    voc_stft = hdf5_file["voc_stft"][()]

                    back_stft = hdf5_file["back_stft"][()]
                else:
                    stft = hdf5_file["voc_stft"][()]
                    # stft = np.clip(stft, 0.0, 1.0)

            speaker_name = voc_to_open.split('_')[1]
            speaker_index = config.singers.index(speaker_name)

            mel = (mel - min_feat)/(max_feat-min_feat)

            


            for j in range(config.samples_per_file):
                voc_idx = np.random.randint(0,len(mel)-config.max_phr_len)
                feats_targs.append(mel[voc_idx:voc_idx+config.max_phr_len])
                if config.mix_aug:

                    gain_back = np.clip(np.random.rand(1), 0.0, 1.0)

                    gain_voc = np.clip(np.random.rand(1), 0.5,1.0)

                    mix_stft = (voc_stft[voc_idx:voc_idx+config.max_phr_len,:]*gain_voc + back_stft[voc_idx:voc_idx+config.max_phr_len,:]*gain_back)*(gain_back+gain_voc)

                    stft_targs.append(mix_stft)
                else:

                    stft_targs.append(stft[voc_idx:voc_idx+config.max_phr_len])

                targets_speakers.append(speaker_index)


        feats_targs = np.array(feats_targs)

        stft_targs = np.clip(np.array(stft_targs), 0.0, 1.0)

        assert feats_targs.max()<=1.0 and feats_targs.min()>=0.0

        yield feats_targs, stft_targs, np.array(targets_speakers)

def get_stats():
    if config.use_casas:

        voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and (x.startswith('yam') or x.startswith('casas'))]

    else:

        voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('yam')]


    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000

    count = 0

    too_small = []
 

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

        # import pdb;pdb.set_trace()

        feats = voc_file["world_feats"][()]

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])


        f0[f0==0] = med



        feats[:,-2] = f0

        if len(feats)<=config.max_phr_len:
            too_small.append(voc_to_open)



        maxi_voc_feat = np.array(feats).max(axis=0)

        for i in range(len(maxi_voc_feat)):
            if maxi_voc_feat[i]>max_feat[i]:
                max_feat[i] = maxi_voc_feat[i]

        mini_voc_feat = np.array(feats).min(axis=0)

        for i in range(len(mini_voc_feat)):
            if mini_voc_feat[i]<min_feat[i]:
                min_feat[i] = mini_voc_feat[i] 
        count+=1

        utils.progress(count, len(voc_list), "Processed")  


    if config.use_casas:

        hdf5_file = h5py.File('./stats.hdf5', mode='w')
    else:
        hdf5_file = h5py.File('./stats_yam.hdf5', mode='w')

    hdf5_file.create_dataset("feats_maximus", [66], np.float32) 
    hdf5_file.create_dataset("feats_minimus", [66], np.float32)   


    hdf5_file["feats_maximus"][:] = max_feat
    hdf5_file["feats_minimus"][:] = min_feat


    hdf5_file.close()

    import pdb;pdb.set_trace()






def main():
    # gen_train_val()
    # get_stats()
    gen = data_gen_npss('Train')
    while True :
        start_time = time.time()
        feats_targs,bppbpp, targets_singers = next(gen)
        print(time.time()-start_time)



        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()
