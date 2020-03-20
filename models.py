import tensorflow as tf
import modules_tf as modules
import config
from data_pipeline import data_gen, data_gen_stft, data_gen_mask, data_gen_chain, SATBBatchGenerator
import time, os
import utils
import h5py
import numpy as np
import mir_eval
import pandas as pd
from random import randint
import librosa
import functools
import sig_process
import vamp_notes
import midi_process

import soundfile as sf

import matplotlib.pyplot as plt
from scipy.ndimage import filters
from tensorflow.contrib.signal.python.ops import window_ops

window = functools.partial(window_ops.hann_window, periodic=True)

def binary_cross(p,q):
    return -(p * tf.log(q + 1e-12) + (1 - p) * tf.log( 1 - q + 1e-12))

class Model(object):
    def __init__(self):
        self.get_placeholders()
        self.model()


    def test_file_all(self, file_name, sess):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        return scores

    def validate_file(self, file_name, sess):
        """
        Function to extract multi pitch from file, for validation. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        pre = scores['Precision']
        acc = scores['Accuracy']
        rec = scores['Recall']
        return pre, acc, rec




    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


        sess.run(self.init_op)

        ckpt = tf.train.get_checkpoint_state(log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)


    def save_model(self, sess, epoch, log_dir):
        """
        Save the model.
        """
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        self.saver.save(sess, checkpoint_file, global_step=epoch)

    def print_summary(self, print_dict, epoch, duration):
        """
        Print training summary to console, every N epochs.
        Summary will depend on model_mode.
        """

        print('epoch %d took (%.3f sec)' % (epoch + 1, duration))
        for key, value in print_dict.items():
            print('{} : {}'.format(key, value))
            

class AutoVC(Model):

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """

        self.optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.optimizer.minimize(self.final_loss, global_step = self.global_step)


    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """

        self.recon_loss = tf.reduce_sum(tf.square(self.input_placeholder - self.output) ) 

        self.content_loss = tf.reduce_sum(tf.abs(self.content_embedding_1 - self.content_embedding_2))

        self.recon_loss_0 = tf.reduce_sum(tf.square(self.input_placeholder - self.output_1))



        self.final_loss = self.recon_loss + config.mu * self.recon_loss_0 + config.lamda * self.content_loss



    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """


        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)

        self.recon_summary = tf.summary.scalar('recon_loss', self.recon_loss)

        self.recon_0_summary = tf.summary.scalar('recon_0_loss', self.recon_loss_0)

        self.content_summary = tf.summary.scalar('content_loss', self.content_loss)

        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()


    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.output_features),
                                           name='input_placeholder')       


        self.speaker_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        self.speaker_onehot_labels = tf.one_hot(indices=tf.cast(self.speaker_labels, tf.int32), depth = config.num_speakers)

        self.speaker_labels_1 = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        self.speaker_onehot_labels_1 = tf.one_hot(indices=tf.cast(self.speaker_labels_1, tf.int32), depth = config.num_speakers)

        self.is_train = tf.placeholder(tf.bool, name="is_train")


    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()


        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir)
        self.get_summary(sess, config.log_dir)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):

            data_generator = data_gen_vc()
            val_generator = data_gen_vc(mode = 'Val')
            

            epoch_final_loss = 0
            epoch_recon_loss = 0
            epoch_recon_0_loss = 0
            epoch_content_loss = 0

            val_final_loss = 0
            val_recon_loss = 0
            val_recon_0_loss = 0
            val_content_loss = 0

            batch_num = 0

            start_time = time.time()

            with tf.variable_scope('Training'):
                for feats_targs, targets_speakers in data_generator:


                    final_loss, recon_loss, recon_loss_0, content_loss,  summary_str = self.train_model(feats_targs[:,:,:64], targets_speakers, sess)

                    epoch_final_loss+=final_loss
                    epoch_recon_loss+=recon_loss
                    epoch_recon_0_loss+=recon_loss_0
                    epoch_content_loss+=content_loss

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num
                epoch_recon_loss = epoch_recon_loss/batch_num
                epoch_recon_0_loss = epoch_recon_0_loss/batch_num
                epoch_content_loss = epoch_content_loss/batch_num

                print_dict = {"Final Loss": epoch_final_loss}

                print_dict["Recon Loss"] =  epoch_recon_loss
                print_dict["Recon Loss_0 "] =  epoch_recon_0_loss
                print_dict["Content Loss"] =  epoch_content_loss



            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for feats_targs, targets_speakers in val_generator:


                        final_loss, recon_loss, recon_loss_0, content_loss,  summary_str = self.validate_model(feats_targs[:,:,:64], targets_speakers, sess)

                        val_final_loss+=final_loss
                        val_recon_loss+=recon_loss
                        val_recon_0_loss+=recon_loss_0
                        val_content_loss+=content_loss

                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()

                        utils.progress(batch_num,config.batches_per_epoch_val, suffix = 'validation done')

                        batch_num+=1

                    val_final_loss = val_final_loss/batch_num
                    val_recon_loss = val_recon_loss/batch_num
                    val_recon_0_loss = val_recon_0_loss/batch_num
                    val_content_loss = val_content_loss/batch_num

                    print_dict["Val Final Loss"] = val_final_loss

                    print_dict["Val Recon Loss"] =  val_recon_loss
                    print_dict["Val Recon Loss_0 "] =  val_recon_0_loss
                    print_dict["Val Content Loss"] =  val_content_loss



            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)


    def train_model(self,feats_targs, targets_speakers, sess):
        """
        Function to train the model for each epoch
        """


        feed_dict = {self.input_placeholder: feats_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers,  self.is_train: True}

            
        _, final_loss, recon_loss, recon_loss_0, content_loss = sess.run([self.final_train_function,self.final_loss, self.recon_loss, self.recon_loss_0, self.content_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, recon_loss, recon_loss_0, content_loss, summary_str
 

    def validate_model(self,feats_targs, targets_speakers, sess):
        """
        Function to train the model for each epoch
        """


        feed_dict = {self.input_placeholder: feats_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers,  self.is_train: False}

            
        final_loss, recon_loss, recon_loss_0, content_loss = sess.run([self.final_loss, self.recon_loss, self.recon_loss_0, self.content_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, recon_loss, recon_loss_0, content_loss, summary_str



    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        # if file_name.endswith('.hdf5'):

        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = stat_file["feats_maximus"][()]
        min_feat = stat_file["feats_minimus"][()]


        stat_file.close()  

        with h5py.File(config.voice_dir+file_name, "r") as hdf5_file:
            if config.mode == "voc":
                mel = hdf5_file["world_feats"][()]
            elif config.mode == "ori":
                mel = hdf5_file["mel_stft"][()]

        

        return mel

    def read_wav_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs)

        audio = np.float64(audio)

        if len(audio.shape) == 2:

            vocals = np.array((audio[:,1]+audio[:,0])/2)

        else: 
            vocals = np.array(audio)

        voc_stft = abs(utils.stft(vocals))

        feats = utils.stft_to_feats(vocals,fs)

        voc_stft = np.clip(voc_stft, 0.0, 1.0)

        return voc_stft, feats



    def test_file_wav(self, file_name, speaker_index):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir =  config.log_dir)
        mel = self.read_wav_file(file_name)

        out_mel = self.process_file(mel, speaker_index, sess)

        self.plot_features(mel, out_mel)



    def test_file_hdf5(self, file_name, speaker_index, speaker_index_2):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir =  config.log_dir)
        mel = self.read_hdf5_file(file_name)

        if config.mode == "voc":

            out_mel = self.process_file(mel[:,:-2], speaker_index, speaker_index_2, sess)

        elif config.mode == "ori": 

            out_mel = self.process_file(mel, speaker_index, speaker_index_2, sess)


        self.plot_features(mel, out_mel)





        synth = utils.query_yes_no("Synthesize output? ")

        if synth:
            gen_change = utils.query_yes_no("Change in gender? ")
            if gen_change:
                female_male = utils.query_yes_no("Female to male?")
                if female_male:
                    out_featss = np.concatenate((out_mel[:mel.shape[0]], mel[:out_mel.shape[0],-2:-1]-12, mel[:out_mel.shape[0],-1:]), axis = -1)
                else:
                    out_featss = np.concatenate((out_mel[:mel.shape[0]], mel[:out_mel.shape[0],-2:-1]+12, mel[:out_mel.shape[0],-1:]), axis = -1)
            else:
                out_featss = np.concatenate((out_mel[:mel.shape[0]], mel[:out_mel.shape[0],-2:]), axis = -1)

            audio_out = utils.feats_to_audio(out_featss) 

            sf.write('./{}_{}.wav'.format(file_name[:-5], config.singers[speaker_index_2]), audio_out, config.fs)

        synth_ori = utils.query_yes_no("Synthesize ground truth with vocoder? ")

        if synth_ori:
            audio = utils.feats_to_audio(mel) 
            sf.write('./{}_{}_ori.wav'.format(file_name[:-5], config.singers[speaker_index]), audio, config.fs)


    def plot_features(self, feats, out_feats):

        plt.figure(1)
        
        ax1 = plt.subplot(211)

        plt.imshow(feats[:,:60].T,aspect='auto',origin='lower')

        ax1.set_title("Ground Truth STFT", fontsize=10)

        ax3 =plt.subplot(212, sharex = ax1, sharey = ax1)

        ax3.set_title("Output STFT", fontsize=10)

        plt.imshow(out_feats[:,:60].T,aspect='auto',origin='lower')

        plt.show()


    def process_file(self, mel, speaker_index, speaker_index_2, sess):

        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = stat_file["feats_maximus"][()]
        min_feat = stat_file["feats_minimus"][()]


        stat_file.close()  
        if config.mode == "voc":

            mel = (mel - min_feat[:-2])/(max_feat[:-2]-min_feat[:-2])

        in_batches_mel, nchunks_in = utils.generate_overlapadd(mel)

        out_batches_mel = []

        for in_batch_mel in in_batches_mel :
            speaker = np.repeat(speaker_index, config.batch_size)
            speaker_2 = np.repeat(speaker_index_2, config.batch_size)
            feed_dict = {self.input_placeholder: in_batch_mel, self.speaker_labels:speaker,self.speaker_labels_1:speaker_2, self.is_train: False}
            mel = sess.run(self.output, feed_dict=feed_dict)

            out_batches_mel.append(mel)
        out_batches_mel = np.array(out_batches_mel)

        out_batches_mel = utils.overlapadd(out_batches_mel,nchunks_in)

        if config.mode == "voc":

            out_batches_mel = out_batches_mel*(max_feat[:-2] - min_feat[:-2]) + min_feat[:-2]

        return out_batches_mel



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """


        with tf.variable_scope('encoder') as scope:
            self.content_embedding_1 = modules.content_encoder(self.input_placeholder, self.speaker_onehot_labels, self.is_train)

        if config.mode == "voc":
            with tf.variable_scope('decoder') as scope: 
                self.output_1 = modules.decoder(self.content_embedding_1, self.speaker_onehot_labels_1, self.is_train)
        elif config.mode == "ori":
            with tf.variable_scope('decoder') as scope: 
                self.output_1 = modules.decoder(self.content_embedding_1, self.speaker_onehot_labels, self.is_train)
        with tf.variable_scope('post_net') as scope: 
            self.residual = modules.post_net(self.output_1, self.is_train)
            self.output = self.output_1 + self.residual
        with tf.variable_scope('encoder') as scope:
            scope.reuse_variables()
            self.content_embedding_2 = modules.content_encoder(self.output, self.speaker_onehot_labels, self.is_train)

class MaskSep(Model):

    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        if config.mask_emb:
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            auto_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

            self.auto_saver = tf.train.Saver(max_to_keep= config.max_models_to_keep, var_list = auto_var_list)

            # if not self.stft_var_list:
            self.stft_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_encoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_post_net') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'F0_Model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Vuv_Model')

            self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep, var_list = self.stft_var_list)

            sess.run(self.init_op)

            ckpt_auto = tf.train.get_checkpoint_state(config.log_dir_content)

            if ckpt_auto and ckpt_auto.model_checkpoint_path:
                print("Using the AUTOVC model in %s"%ckpt_auto.model_checkpoint_path)
                self.auto_saver.restore(sess, ckpt_auto.model_checkpoint_path)

            ckpt_stft = tf.train.get_checkpoint_state(config.log_dir)

            if ckpt_stft and ckpt_stft.model_checkpoint_path:
                print("Using the STFT model in %s"%ckpt_stft.model_checkpoint_path)
                self.saver.restore(sess, ckpt_stft.model_checkpoint_path)
            else:
                print("Trining for folder {}".format(config.log_dir))
        else:
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


            sess.run(self.init_op)

            ckpt = tf.train.get_checkpoint_state(config.log_dir)

            if ckpt and ckpt.model_checkpoint_path:
                print("Using the model in %s"%ckpt.model_checkpoint_path)
                self.saver.restore(sess, ckpt.model_checkpoint_path)


    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """
        if config.mask_emb:
            self.mask_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_encoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_post_net')
        else:
            self.mask_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Mask_Model')


        self.mask_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)


        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.mask_train_function = self.mask_optimizer.minimize(self.final_loss, global_step = self.global_step, var_list = self.mask_params)


    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """


        if config.mul_mask:
            soprano = tf.abs(tf.contrib.signal.stft(tf.squeeze(self.soprano_stft_placeholder), frame_length=1024, frame_step=config.hopsize, fft_length=1024, window_fn=window))
            alto = tf.abs(tf.contrib.signal.stft(tf.squeeze(self.alto_stft_placeholder), frame_length=1024, frame_step=config.hopsize, fft_length=1024, window_fn=window))
            tenor = tf.abs(tf.contrib.signal.stft(tf.squeeze(self.tenor_stft_placeholder), frame_length=1024, frame_step=config.hopsize, fft_length=1024, window_fn=window))
            bass = tf.abs(tf.contrib.signal.stft(tf.squeeze(self.bass_stft_placeholder), frame_length=1024, frame_step=config.hopsize, fft_length=1024, window_fn=window))

            self.soprano_loss = tf.reduce_sum(tf.abs(self.mask[:,:,:config.output_features]*self.mix - soprano))
            self.alto_loss = tf.reduce_sum(tf.abs(self.mask[:,:,config.output_features: config.output_features*2]*self.mix - alto))
            self.tenor_loss = tf.reduce_sum(tf.abs(self.mask[:,:,config.output_features*2:]*self.mix - tenor))
            self.bass_loss = tf.reduce_sum(tf.abs((1-(self.mask[:,:,:config.output_features] + self.mask[:,:,config.output_features: config.output_features*2] + self.mask[:,:,config.output_features*2:]))*self.mix - bass))

        else:
            self.voc_loss = tf.reduce_sum(tf.abs(self.mask - self.voc_stft_placeholder))
            self.back_loss = tf.reduce_sum(tf.abs((self.input_placeholder-self.mask) - self.back_stft_placeholder))

        self.recon_loss = self.soprano_loss + self.alto_loss + self.tenor_loss + self.bass_loss 

        if config.mask_emb:

            if config.mul_mask:
                self.voc_loss_1 = tf.reduce_sum(tf.abs(tf.sigmoid(self.output_stft_1)*self.input_placeholder - self.voc_stft_placeholder))
                self.back_loss_1 = tf.reduce_sum(tf.abs((1-tf.sigmoid(self.output_stft_1))*self.input_placeholder - self.back_stft_placeholder))

            else:
                self.voc_loss_1 = tf.reduce_sum(tf.abs(tf.sigmoid(self.output_stft_1) - self.voc_stft_placeholder))
                self.back_loss_1 = tf.reduce_sum(tf.abs((self.input_placeholder-tf.sigmoid(self.output_stft_1)) - self.back_stft_placeholder))

            self.recon_loss_1 = self.voc_loss_1 + self.back_loss_1 

            self.content_loss = tf.reduce_sum(tf.abs(self.content_embedding_ori - self.content_embedding_stft))

            self.content_loss_2 = tf.reduce_sum(tf.abs(self.content_embedding_ori - self.content_embedding_stft_2))

            self.content_loss = self.content_loss + self.content_loss_2

            self.final_loss = self.recon_loss + config.mu * self.recon_loss_1 + config.lamda * self.content_loss

        else:
            self.final_loss = self.recon_loss

        



    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """
        if config.mask_emb:

            self.voc_summary = tf.summary.scalar('voc_loss', self.voc_loss)

            self.back_summary = tf.summary.scalar('back_loss', self.back_loss)

            self.recon_summary = tf.summary.scalar('recon_loss', self.recon_loss)

            self.recon_0_summary = tf.summary.scalar('recon_0_loss', self.recon_loss_1)

            self.content_summary = tf.summary.scalar('content_loss', self.content_loss)

        self.loss_summary = tf.summary.scalar('loss', self.final_loss)

        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, (config.max_phr_len -1)* config.hopsize+1024,1),name='input_placeholder')
        self.soprano_stft_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,(config.max_phr_len -1)* config.hopsize+1024, 1),name='soprano_stft_placeholder')
        self.alto_stft_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,(config.max_phr_len -1)* config.hopsize+1024, 1),name='alto_stft_placeholder')
        self.tenor_stft_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,(config.max_phr_len -1)* config.hopsize+1024, 1),name='tenor_stft_placeholder')
        self.bass_stft_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,(config.max_phr_len -1)* config.hopsize+1024,1 ),name='bass_stft_placeholder')
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        self.feats_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.input_features),
                                           name='feats_placeholder')       

        self.speaker_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        self.speaker_onehot_labels = tf.one_hot(indices=tf.cast(self.speaker_labels, tf.int32), depth = config.num_speakers)


    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()

        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir)
        self.get_summary(sess, config.log_dir)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):
            data_generator = SATBBatchGenerator("../Darius/Wave-U-Net/satb_dataset.hdf5")
            val_generator = SATBBatchGenerator("../Darius/Wave-U-Net/satb_dataset.hdf5",partition='valid')
            start_time = time.time()


            batch_num = 0
            epoch_final_loss = 0
            epoch_voc_loss = 0
            epoch_back_loss = 0


            val_final_loss = 0
            val_voc_loss = 0
            val_back_loss = 0


            with tf.variable_scope('Training'):

                for _ in range(config.batches_per_epoch_train):
                    batch = next(data_generator)

                    loss, summary_str = self.train_model(batch, sess)

                    epoch_final_loss+=loss


                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num
                epoch_voc_loss = epoch_voc_loss/batch_num
                epoch_back_loss = epoch_back_loss/batch_num

                print_dict = {"Final Loss": epoch_final_loss}
                print_dict["Vocal Loss"] =  epoch_voc_loss
                print_dict["Back Loss"] =  epoch_back_loss

            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for _ in range(config.batches_per_epoch_val):
                        batch = next(val_generator)

                        loss, summary_str = self.validate_model(batch, sess)
                        val_final_loss+=loss


                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_final_loss = val_final_loss/batch_num
                    val_voc_loss = val_voc_loss/batch_num
                    val_back_loss = val_back_loss/batch_num


                    print_dict["Val Final Loss"] =  val_final_loss
                    print_dict["Val Vocal Loss"] =  val_voc_loss
                    print_dict["Val Back Loss"] =  val_back_loss

            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self,batch, sess):
        """
        Function to train the model for each epoch
        """

        if config.mask_emb:
            if config.use_f0_emb:
                feed_dict = {self.input_placeholder: mix, self.voc_stft_placeholder: voc, self.back_stft_placeholder: back, self.feats_placeholder:feats, self.speaker_labels:speakers, self.is_train: True}

            else:
                feed_dict = {self.input_placeholder: mix, self.voc_stft_placeholder: voc, self.back_stft_placeholder: back, self.feats_placeholder:feats[:,:,:64], self.speaker_labels:speakers, self.is_train: True}
        else:


            
            feed_dict = {self.input_placeholder: batch['mix'], self.soprano_stft_placeholder: batch['soprano'], self.alto_stft_placeholder: batch['alto'],\
            self.tenor_stft_placeholder: batch['tenor'], self.bass_stft_placeholder: batch['bass'], self.is_train: True}


        _, final_loss= sess.run([self.mask_train_function, self.final_loss], feed_dict=feed_dict)


        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, summary_str

    def validate_model(self, batch, sess):
        """
        Function to train the model for each epoch
        """
        if config.mask_emb:
            if config.use_f0_emb:
                feed_dict = {self.input_placeholder: mix, self.voc_stft_placeholder: voc, self.back_stft_placeholder: back, self.feats_placeholder:feats, self.speaker_labels:speakers, self.is_train: False}

            else:
                feed_dict = {self.input_placeholder: mix, self.voc_stft_placeholder: voc, self.back_stft_placeholder: back, self.feats_placeholder:feats[:,:,:64], self.speaker_labels:speakers, self.is_train: False}
        else:            
            feed_dict = {self.input_placeholder: batch['mix'], self.soprano_stft_placeholder: batch['soprano'], self.alto_stft_placeholder: batch['alto'],\
            self.tenor_stft_placeholder: batch['tenor'], self.bass_stft_placeholder: batch['bass'], self.is_train: False}


        final_loss = sess.run(self.final_loss, feed_dict=feed_dict)


        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return final_loss, summary_str


    def read_wav_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs)

        audio = np.float64(audio)

        if len(audio.shape) == 2:

            vocals = np.array((audio[:,1]+audio[:,0])/2)

        else: 
            vocals = np.array(audio)

        # 


        return vocals


    def test_file_wav(self, file_name, acap_file=None):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir =  config.log_dir)
        audio = self.read_wav_file(file_name)

        mix_stft = utils.stft(audio, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        abs_mix_stft = abs(mix_stft)

        pha_mix_stft = np.angle(mix_stft)

        if acap_file:
            voc_stft = self.read_wav_file(acap_file)
            abs_voc_stft = abs(voc_stft)
            pha_voc_stft = np.angle(voc_stft)
            audio_in = utils.istft(abs_voc_stft, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)
            audio_back = utils.istft(abs_mix_stft - abs_voc_stft, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        out_soprano, out_alto, out_tenor, out_bass = self.process_file(audio, sess)

        import pdb;pdb.set_trace()

        if config.use_griff:
            audio_out = utils.griffinlim(out_voc)
            if acap_file:
                audio_in = utils.griffinlim(abs_voc_stft)
        else:
            audio_out = utils.istft(out_bass[:pha_mix_stft.shape[0]], pha_mix_stft[:out_soprano.shape[0]], hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        audio_mix_out = utils.istft(abs_mix_stft, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        audio_back_out = utils.istft(out_back, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        

        plt.figure(1)

        if acap_file:

            ax1 = plt.subplot(311)

            plt.imshow(np.log(abs_mix_stft.T),aspect='auto',origin='lower')

            ax1.set_title("Input STFT", fontsize=10)

            ax2 = plt.subplot(312, sharex = ax1, sharey = ax1)

            plt.imshow(np.log(abs_voc_stft.T),aspect='auto',origin='lower')

            ax2.set_title("Ground Truth Vocoder Features", fontsize=10)

            ax3 = plt.subplot(313, sharex = ax1, sharey = ax1)

            plt.imshow(np.log(out_voc.T),aspect='auto',origin='lower')

            ax3.set_title("Output Vocoder Features", fontsize=10)

            plt.show()

        else:


            ax1 = plt.subplot(211)

            plt.imshow(np.log(abs_mix_stft.T),aspect='auto',origin='lower')

            ax1.set_title("Input STFT", fontsize=10)

            ax3 = plt.subplot(212, sharex = ax1, sharey = ax1)

            plt.imshow(np.log(out_voc.T),aspect='auto',origin='lower')

            ax3.set_title("Output Vocoder Features", fontsize=10)

            plt.show()

        if config.mul_mask:

            if config.use_griff:

                sf.write('./{}_output_voc_griff.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)
            else:
                sf.write('./{}_output_voc_mask.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)

            sf.write('./{}_output_back_mask.wav'.format(file_name.split('/')[-1][:-4]), audio_back_out, config.fs)
        else:
            sf.write('./{}_output_voc.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)

            sf.write('./{}_output_back.wav'.format(file_name.split('/')[-1][:-4]), audio_back_out, config.fs)

        sf.write('./{}_mix.wav'.format(file_name.split('/')[-1][:-4]), audio_mix_out, config.fs)

        if acap_file:

            sf.write('./{}_ori.wav'.format(file_name.split('/')[-1][:-4]), audio_in, config.fs)
            sf.write('./{}_back.wav'.format(file_name.split('/')[-1][:-4]), audio_back, config.fs)


    def process_file(self, audio, sess):

        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        in_batches_stft, nchunks_in = utils.generate_overlapadd(np.expand_dims(audio, -1), time_context=(config.max_phr_len -1)* config.hopsize+1024, overlap=((config.max_phr_len -1)* config.hopsize+1024)/2)

        out_batches_soprano = []
        out_batches_alto = []
        out_batches_tenor = []
        out_batches_bass = []

        for in_batch_stft in in_batches_stft :
            feed_dict = {self.input_placeholder: in_batch_stft, self.is_train: False}
            out_soprano, out_alto, out_tenor, out_bass = sess.run([self.mask[:,:,:config.output_features]*self.mix,self.mask[:,:,config.output_features: config.output_features*2]*self.mix,\
              self.mask[:,:,config.output_features*2:]*self.mix,(1-(self.mask[:,:,:config.output_features] + self.mask[:,:,config.output_features: config.output_features*2] + self.mask[:,:,config.output_features*2:]))*self.mix], feed_dict=feed_dict)
            out_batches_soprano.append(out_soprano)
            out_batches_alto.append(out_alto)
            out_batches_tenor.append(out_tenor)
            out_batches_bass.append(out_bass)

        out_batches_soprano = np.array(out_batches_soprano)
        out_batches_alto = np.array(out_batches_alto)
        out_batches_tenor = np.array(out_batches_tenor)
        out_batches_bass = np.array(out_batches_bass)

        out_soprano = utils.overlapadd(out_batches_soprano,nchunks_in)
        out_alto = utils.overlapadd(out_batches_alto,nchunks_in)
        out_tenor = utils.overlapadd(out_batches_tenor,nchunks_in)
        out_bass = utils.overlapadd(out_batches_bass,nchunks_in)

        return out_soprano, out_alto, out_tenor, out_bass

    def read_med_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs, mono=False)

        audio = np.float64(audio)

        vocals = np.array(audio[1,:])

        mixture = np.array(audio[0,:])

        voc_stft = utils.stft(mixture, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        input_stft = np.clip(abs(voc_stft), 0.0, 1.0)

        angle = np.angle(voc_stft)

        return input_stft, angle

    def test_folder_wav(self, folder_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir=config.log_dir)

        file_list = [x for x in os.listdir(folder_name) if x in config.med_to_use]

        count = 0

        unprocessable = []

        for file_name in file_list:
            try:
                mel, pha_mix_stft = self.read_med_file(os.path.join(folder_name, file_name))
                out_mel, out_back= self.process_file(mel, sess)
                audio_out = utils.istft(out_mel, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)
                sf.write(os.path.join(config.output_dir,'./{}_output_mask.wav'.format(file_name.split('/')[-1][:-4])), audio_out, config.fs)


            except:
                unprocessable.append(file_name)

            count+=1

            utils.progress(count, len(file_list), "Files processed")

        print(unprocessable)

    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """
        if config.mask_emb:
            with tf.variable_scope('encoder') as scope:
                self.content_embedding_ori = modules.content_encoder(self.feats_placeholder, self.speaker_onehot_labels, self.is_train)

            with tf.variable_scope('stft_encoder') as scope:
                self.content_embedding_stft = modules.content_encoder_sep(self.input_placeholder, self.is_train)

            with tf.variable_scope('stft_decoder') as scope: 
                self.output_stft_1 = modules.decoder_sep(self.content_embedding_stft, self.input_placeholder, self.is_train)


            with tf.variable_scope('stft_post_net') as scope: 

                self.residual_stft = modules.post_net_sep(self.output_stft_1, self.is_train)
                self.mask = tf.sigmoid(self.output_stft_1 + self.residual_stft)
                if config.mul_mask:
                    self.output_stft = self.input_placeholder * self.mask
                else:
                    self.output_stft = self.mask

            with tf.variable_scope('stft_encoder') as scope:
                scope.reuse_variables()
                self.content_embedding_stft_2 = modules.content_encoder_sep(self.output_stft, self.is_train)


        else:
            with tf.variable_scope('Mask_Model') as scope:
                self.mask = modules.enc_dec_mask(self.input_placeholder, self.is_train)


class Chain(Model):


    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """
        self.mask_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Mask_Model')

        self.harm_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Harm_Model')
        self.f0_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'F0_Model')
        self.vuv_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Vuv_Model')

        self.voc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Voc_Model')
        self.back_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Back_Model')


        self.mask_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.harm_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.f0_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.vuv_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.voc_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.back_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_f0 = tf.Variable(0, name='global_step_f0', trainable=False)
        self.global_step_vuv = tf.Variable(0, name='global_step_vuv', trainable=False)
        self.global_step_harm = tf.Variable(0, name='global_step_harm', trainable=False)
        self.global_step_voc = tf.Variable(0, name='global_step_voc', trainable=False)
        self.global_step_back = tf.Variable(0, name='global_step_back', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.mask_train_function = self.mask_optimizer.minimize(self.mask_loss, global_step = self.global_step, var_list = self.mask_params)
            self.harm_train_function = self.harm_optimizer.minimize(self.harm_loss, global_step = self.global_step_harm, var_list = self.harm_params)
            self.f0_train_function = self.f0_optimizer.minimize(self.f0_loss, global_step = self.global_step_f0, var_list = self.f0_params)
            self.vuv_train_function = self.vuv_optimizer.minimize(self.vuv_loss, global_step = self.global_step_vuv, var_list = self.vuv_params)
            self.voc_train_function = self.voc_optimizer.minimize(self.voc_loss, global_step = self.global_step_voc, var_list = self.voc_params)
            self.back_train_function = self.back_optimizer.minimize(self.back_loss, global_step = self.global_step_back, var_list = self.back_params)


    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """


        self.mask_loss = tf.reduce_sum(tf.abs(self.mask*self.input_placeholder - self.voc_stft_placeholder)) + tf.reduce_sum(tf.abs((1-self.mask)*self.input_placeholder - self.back_stft_placeholder))

        self.harm_loss = tf.reduce_sum(tf.abs(self.harm - self.harm_placeholder)) + tf.reduce_sum(tf.abs(self.ap - self.ap_placeholder))

        self.vuv_loss = tf.reduce_mean(tf.reduce_mean(binary_cross(self.vuv_placeholder, self.vuv)))

        self.f0_loss = tf.reduce_sum(tf.abs(self.f0 - self.f0_placeholder)*(1-self.vuv_placeholder)) 

        self.voc_loss = tf.reduce_sum(tf.abs(self.vocals_out - self.voc_stft_placeholder))

        self.back_loss = tf.reduce_sum(tf.abs(self.back_out - self.back_stft_placeholder))




    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """

        self.voc_summary = tf.summary.scalar('voc_loss', self.voc_loss)

        self.back_summary = tf.summary.scalar('back_loss', self.back_loss)

        self.mask_summary = tf.summary.scalar('mask_loss', self.mask_loss)


        self.harm_summary = tf.summary.scalar('harm_loss', self.harm_loss)

        self.f0_summary = tf.summary.scalar('f0_loss', self.f0_loss)

        self.vuv_summary = tf.summary.scalar('vuv_loss', self.vuv_loss)


        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.stft_features),name='input_placeholder')
        self.voc_stft_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.stft_features),name='voc_stft_placeholder')
        self.back_stft_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.stft_features),name='back_stft_placeholder')
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self.harm_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,60),name='harm_placeholder')
        self.ap_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,4),name='ap_placeholder')

        self.f0_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='f0_placeholder')

        self.vuv_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='vuv_placeholder')

    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()

        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir)
        self.get_summary(sess, config.log_dir)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):
            data_generator = data_gen_chain()
            val_generator = data_gen_chain(mode = 'Val')
            start_time = time.time()


            batch_num = 0
            epoch_final_loss = 0
            epoch_voc_loss = 0
            epoch_back_loss = 0


            val_final_loss = 0
            val_voc_loss = 0
            val_back_loss = 0


            with tf.variable_scope('Training'):

                for mix, voc, back, feats in data_generator:

                    mask_loss, voc_loss, back_loss,  summary_str = self.train_model(mix, voc, back, feats, sess)

                    epoch_final_loss+=mask_loss
                    epoch_voc_loss+=voc_loss
                    epoch_back_loss+=back_loss

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num
                epoch_voc_loss = epoch_voc_loss/batch_num
                epoch_back_loss = epoch_back_loss/batch_num

                print_dict = {"Final Loss": epoch_final_loss}
                print_dict["Vocal Loss"] =  epoch_voc_loss
                print_dict["Back Loss"] =  epoch_back_loss

            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for mix, voc, back, feats in val_generator:

                        loss, voc_loss, back_loss,  summary_str = self.validate_model(mix, voc, back, feats, sess)
                        val_final_loss+=loss
                        val_voc_loss+=voc_loss
                        val_back_loss+=back_loss

                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_final_loss = val_final_loss/batch_num
                    val_voc_loss = val_voc_loss/batch_num
                    val_back_loss = val_back_loss/batch_num


                    print_dict["Val Final Loss"] =  val_final_loss
                    print_dict["Val Vocal Loss"] =  val_voc_loss
                    print_dict["Val Back Loss"] =  val_back_loss

            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self, mix, voc, back, feat, sess):
        """
        Function to train the model for each epoch
        """

        feed_dict = {self.input_placeholder: mix, self.voc_stft_placeholder: voc, self.back_stft_placeholder: back,  self.harm_placeholder: feat[:,:,:60], self.ap_placeholder: feat[:,:,60:64], \
            self.f0_placeholder: feat[:,:,-2:-1], self.vuv_placeholder: feat[:,:,-1:], self.is_train: True}

        _,_,_,_, _,_, mask_loss, voc_loss, back_loss = sess.run([self.mask_train_function, self.harm_train_function, self.f0_train_function, self.vuv_train_function, self.voc_train_function, self.back_train_function,\
         self.mask_loss, self.voc_loss, self.back_loss], feed_dict=feed_dict)


        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return mask_loss, voc_loss, back_loss, summary_str

    def validate_model(self, mix, voc, back, feat, sess):
        """
        Function to train the model for each epoch
        """
        feed_dict = {self.input_placeholder: mix, self.voc_stft_placeholder: voc, self.back_stft_placeholder: back,  self.harm_placeholder: feat[:,:,:60], self.ap_placeholder: feat[:,:,60:64], \
            self.f0_placeholder: feat[:,:,-2:-1], self.vuv_placeholder: feat[:,:,-1:], self.is_train: False}

        mask_loss, voc_loss, back_loss = sess.run([self.mask_loss, self.voc_loss, self.back_loss], feed_dict=feed_dict)


        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return mask_loss, voc_loss, back_loss, summary_str



    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """

        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')
        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])

        stat_file.close()

        with h5py.File(config.voice_dir + file_name) as voc_file:

            voc_stft = voc_file["voc_stft"][()]

            back_stft = voc_file["back_stft"][()]

            mix_stft = (voc_stft + back_stft)/2

            feats = np.array(voc_file['world_feats'])

            atb = voc_file["atb"][()]

            atb = atb[:, 1:]


        return mix_stft, feats, atb

    def test_file_hdf5(self, file_name):
        """
        Function to extract vocals from hdf5 file.
        """
        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        mix_stft, feats, atb = self.read_hdf5_file(file_name)
        out_feats, out_atb, out_vuv = self.process_file(mix_stft,  sess)

        if config.f0_mode == "discrete":

            est_freq = utils.to_viterbi_cents(out_atb)

            est_freq = est_freq/100
            est_freq = est_freq + 12*np.log2(10) - 12*np.log2(440)
            est_freq = est_freq + 69 

            ori_freq = utils.to_viterbi_cents(atb)

            ori_freq = ori_freq/100
            ori_freq = ori_freq + 12*np.log2(10) - 12*np.log2(440)
            ori_freq = ori_freq + 69

            # import pdb;pdb.set_trace()

        elif config.f0_mode == "cont":
            est_freq = None

        self.plot_features(feats, out_feats, mix_stft, atb, out_atb, est_freq, out_vuv)

        # import pdb;pdb.set_trace()

        synth = utils.query_yes_no("Synthesize output? ")

        if synth:

            if config.f0_mode == "cont":

                audio_out = utils.feats_to_audio(np.concatenate((out_feats[:feats.shape[0],:-1], out_vuv[:feats.shape[0]]) , axis = -1))
                sf.write('./{}_ss_pred.wav'.format(file_name[:-5]), audio_out, config.fs)

            elif config.f0_mode == "discrete":

                audio_out = utils.feats_to_audio(np.concatenate((out_feats[:feats.shape[0]], np.expand_dims(est_freq[:feats.shape[0]],-1), out_vuv[:feats.shape[0]]) , axis = -1))
                sf.write('./{}_ss_pred_dis.wav'.format(file_name[:-5]), audio_out, config.fs)


            # sf.write('./{}_ss_pred.wav'.format(file_name[:-5]), audio_out, config.fs)

        synth = utils.query_yes_no("Synthesize output with original F0? ")

        if synth:

            audio_out = utils.feats_to_audio(np.concatenate((out_feats[:feats.shape[0],:-2], feats[:out_feats.shape[0],-2:]) , axis = -1))

            sf.write('./{}_ss_ori.wav'.format(file_name[:-5]), audio_out, config.fs)

        synth_ori = utils.query_yes_no("Synthesize ground truth with vocoder? ")

        if synth_ori:
            audio = utils.feats_to_audio(feats) 
            sf.write('./{}_ori.wav'.format(file_name[:-5]), audio, config.fs)

    def read_wav_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs)

        audio = np.float64(audio)

        if len(audio.shape) == 2:

            vocals = np.array((audio[:,1]+audio[:,0])/2)

        else: 
            vocals = np.array(audio)

        mix_stft = utils.stft(vocals, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)


        return mix_stft


    def test_file_wav(self, file_name, acap_file=None):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir =  config.log_dir)
        mix_stft = self.read_wav_file(file_name)

        abs_mix_stft = abs(mix_stft)

        pha_mix_stft = np.angle(mix_stft)

        if acap_file:
            voc_stft = self.read_wav_file(acap_file)
            abs_voc_stft = abs(voc_stft)
            pha_voc_stft = np.angle(voc_stft)
            audio_in = utils.istft(abs_voc_stft, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        out_voc, out_back, out_voc_mask, out_back_mask, out_feats = self.process_file(abs_mix_stft, sess)

        audio_out = utils.istft(out_voc, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        audio_out_mask = utils.istft(out_voc_mask, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        audio_mix_out = utils.istft(abs_mix_stft, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        audio_back_out = utils.istft(out_back, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        audio_back_out_mask = utils.istft(out_back_mask, pha_mix_stft, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)

        audio_out_feats = utils.feats_to_audio(out_feats)


        plt.figure(1)

        if acap_file:

            ax1 = plt.subplot(311)

            plt.imshow(np.log(abs_mix_stft.T),aspect='auto',origin='lower')

            ax1.set_title("Input STFT", fontsize=10)

            ax2 = plt.subplot(312, sharex = ax1, sharey = ax1)

            plt.imshow(np.log(abs_voc_stft.T),aspect='auto',origin='lower')

            ax2.set_title("Ground Truth Vocoder Features", fontsize=10)

            ax3 = plt.subplot(313, sharex = ax1, sharey = ax1)

            plt.imshow(np.log(out_voc.T),aspect='auto',origin='lower')

            ax3.set_title("Output Vocoder Features", fontsize=10)

            plt.show()

        else:


            ax1 = plt.subplot(211)

            plt.imshow(np.log(abs_mix_stft.T),aspect='auto',origin='lower')

            ax1.set_title("Input STFT", fontsize=10)

            ax3 = plt.subplot(212, sharex = ax1, sharey = ax1)

            plt.imshow(np.log(out_voc.T),aspect='auto',origin='lower')

            ax3.set_title("Output Vocoder Features", fontsize=10)

            plt.show()



        sf.write('./{}_output_voc_mask.wav'.format(file_name.split('/')[-1][:-4]), audio_out_mask, config.fs)

        sf.write('./{}_output_voc_feats.wav'.format(file_name.split('/')[-1][:-4]), audio_out_feats, config.fs)

        sf.write('./{}_output_back_mask.wav'.format(file_name.split('/')[-1][:-4]), audio_back_out_mask, config.fs)

        sf.write('./{}_output_voc.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)

        sf.write('./{}_output_back.wav'.format(file_name.split('/')[-1][:-4]), audio_back_out, config.fs)

        sf.write('./{}_mix.wav'.format(file_name.split('/')[-1][:-4]), audio_mix_out, config.fs)

        if acap_file:

            sf.write('./{}_ori.wav'.format(file_name.split('/')[-1][:-4]), audio_in, config.fs)




    def plot_features(self, feats, out_feats, mix_stft, atb, out_atb, est_frequ, out_vuv):
        """
        Function to plot output and ground truth features
        """
        plt.figure(1)


        ax1 = plt.subplot(311)

        plt.imshow(np.log(mix_stft.T),aspect='auto',origin='lower')

        ax1.set_title("Input STFT", fontsize=10)

        ax2 = plt.subplot(312, sharex = ax1, sharey = ax1)

        plt.imshow(feats[:,:60].T,aspect='auto',origin='lower')

        ax1.set_title("Ground Truth Vocoder Features", fontsize=10)

        ax3 = plt.subplot(313, sharex = ax1, sharey = ax1)

        ax3.set_title("Output Vocoder Features", fontsize=10)

        plt.imshow(out_feats[:,:60].T,aspect='auto',origin='lower')

        plt.figure(4)


        ax1 = plt.subplot(211)

        plt.plot(feats[:,-1])

        ax1.set_title("Ground Truth VUV", fontsize=10)

        ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)

        plt.plot(out_vuv)

        ax1.set_title("Output VUV", fontsize=10)

        if config.f0_mode == "cont":

            plt.figure(2)
            f0_output = out_feats[:feats.shape[0],-2]
            f0_output = f0_output*(1-feats[:,-1])
            f0_output[f0_output == 0] = np.nan
            plt.plot(f0_output, label = "Predicted Value")
            f0_gt = feats[:,-2]
            f0_gt = f0_gt*(1-feats[:,-1])
            f0_gt[f0_gt == 0] = np.nan
            plt.plot(f0_gt, label="Ground Truth")
            f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
            f0_greater = np.where(f0_difference>config.f0_threshold)
            diff_per = f0_greater[0].shape[0]/len(f0_output)
            plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))

        elif config.f0_mode == "discrete":
            out_atb = out_atb[:atb.shape[0]]
            time_1, ori_freq = utils.process_output(atb)
            time_2, est_freq = utils.process_output(out_atb)
            # import pdb;pdb.set_trace()
            scores = mir_eval.multipitch.evaluate(time_1, ori_freq, time_2, est_freq)
            pre = scores['Precision']
            acc = scores['Accuracy']
            rec = scores['Recall']
            plt.figure(2)
            ax1 = plt.subplot(211)
            plt.imshow(atb.T, origin='lower', aspect='auto')
            ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)
            plt.imshow(out_atb.T, origin='lower', aspect='auto')
            plt.suptitle("Precision: {:.3%}, Accuracy: {:.3%},  Recall: {:.3%}".format(pre, acc, rec))

            plt.figure(3)
            f0_output = est_frequ[:feats.shape[0]]
            f0_output = f0_output*(1-feats[:,-1])
            f0_output[f0_output == 0] = np.nan


            plt.plot(f0_output, label = "Predicted Value")
            f0_gt = feats[:,-2]
            f0_gt = f0_gt*(1-feats[:,-1])
            f0_gt[f0_gt == 0] = np.nan
            plt.plot(f0_gt, label="Ground Truth")
            f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
            f0_greater = np.where(f0_difference>config.f0_threshold)
            diff_per = f0_greater[0].shape[0]/len(f0_output)
            plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))



        plt.show()


    def process_file(self, mix_stft, sess):

        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        in_batches_stft, nchunks_in = utils.generate_overlapadd(mix_stft)

        out_batches_mask = []
        out_batches_voc = []
        out_batches_back = []
        out_batches_feats = []

        for in_batch_stft in in_batches_stft :
            feed_dict = {self.input_placeholder: in_batch_stft, self.is_train: False}
            out_mask, out_voc, out_back, out_harm, out_ap, out_f0, out_vuv = sess.run([self.mask, self.vocals_out, self.back_out, self.harm, self.ap, self.f0, self.vuv], feed_dict=feed_dict)
            out_batches_mask.append(out_mask)
            out_batches_voc.append(out_voc)
            out_batches_back.append(out_back)
            out_batches_feats.append(np.concatenate((out_harm, out_ap, out_f0, np.round(out_vuv)), axis=-1))


        out_batches_mask = np.array(out_batches_mask)

        out_mask = utils.overlapadd(out_batches_mask,nchunks_in)

        out_batches_voc = np.array(out_batches_voc)

        out_voc = utils.overlapadd(out_batches_voc,nchunks_in)

        out_batches_back = np.array(out_batches_back)

        out_back = utils.overlapadd(out_batches_back,nchunks_in)

        out_voc_mask = mix_stft*out_mask[:mix_stft.shape[0]]

        out_back_mask = mix_stft*(1-out_mask[:mix_stft.shape[0]])

        out_feats = utils.overlapadd(np.array(out_batches_feats),nchunks_in)

        out_feats = out_feats[:mix_stft.shape[0]]*(max_feat-min_feat)+min_feat


        return out_voc[:mix_stft.shape[0]], out_back[:mix_stft.shape[0]], out_voc_mask, out_back_mask, out_feats

    def test_folder_wav(self, folder_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir=config.log_dir)

        file_list = [x for x in os.listdir(folder_name) if x.endswith('.wav') and not x.startswith('.')]

        count = 0

        unprocessable = []

        for file_name in file_list:
            try:
                mel = self.read_wav_file(os.path.join(folder_name, file_name))
                out_mel, out_f0, out_vuv = self.process_file(mel, sess)
                out_featss = np.concatenate((out_mel, out_vuv), axis = -1)

                audio_out = utils.feats_to_audio(out_featss) 

                sf.write(os.path.join(config.output_dir,'./{}_output.wav'.format(file_name.split('/')[-1][:-4])), audio_out, config.fs)

                np.save(os.path.join(config.output_dir_np,file_name[:-4]), out_mel)

            except:
                unprocessable.append(file_name)

            count+=1

            utils.progress(count, len(file_list), "Files processed")

    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """
        with tf.variable_scope('Mask_Model') as scope:
                self.mask = modules.enc_dec_mask(self.input_placeholder, self.is_train)

        with tf.variable_scope('Harm_Model') as scope:
            if config.enc_mode == "wave":
                self.harm, self.ap = modules.nr_wavenet(tf.concat([self.input_placeholder, self.mask], axis=-1), self.is_train)
            elif config.enc_mode == "conv":
                self.harm, self.ap = modules.enc_dec(tf.concat([self.input_placeholder, self.mask], axis=-1), self.is_train)
        with tf.variable_scope('F0_Model') as scope:
            if config.enc_mode == "wave":
                self.f0 = modules.nr_wavenet_f0(tf.concat([self.input_placeholder, self.mask], axis=-1), self.harm, self.ap, self.is_train)
            elif config.enc_mode == "conv":
                self.f0 = modules.enc_dec_f0(tf.concat([self.input_placeholder, self.mask], axis=-1), self.harm, self.ap, self.is_train)
        with tf.variable_scope('Vuv_Model') as scope:
            if config.enc_mode == "wave":
                self.vuv = modules.nr_wavenet_vuv(tf.concat([self.input_placeholder, self.mask], axis=-1), self.harm, self.ap, self.f0, self.is_train)
            elif config.enc_mode == "conv":
                self.vuv = modules.enc_dec_vuv(tf.concat([self.input_placeholder, self.mask], axis=-1), self.harm, self.ap, self.f0, self.is_train)
        with tf.variable_scope('Voc_Model') as scope:
            self.vocals_out = modules.enc_dec_out_voc(tf.concat([self.input_placeholder, self.mask, self.harm, self.ap, self.f0, self.vuv], axis=-1), self.is_train)
        with tf.variable_scope('Back_Model') as scope:
            self.back_out = modules.enc_dec_out_voc(tf.concat([self.input_placeholder, self.mask, self.vocals_out, self.harm, self.ap, self.f0, self.vuv], axis=-1), self.is_train)

class SSSynth(Model):

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """
        self.harm_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Harm_Model')
        self.ap_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Ap_Model')
        self.f0_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'F0_Model')
        self.vuv_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Vuv_Model')

        self.harm_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.ap_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.f0_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.vuv_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_ap = tf.Variable(0, name='global_step_ap', trainable=False)
        self.global_step_f0 = tf.Variable(0, name='global_step_f0', trainable=False)
        self.global_step_vuv = tf.Variable(0, name='global_step_vuv', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.harm_train_function = self.harm_optimizer.minimize(self.loss, global_step = self.global_step, var_list = self.harm_params)
            self.f0_train_function = self.f0_optimizer.minimize(self.f0_loss, global_step = self.global_step_f0, var_list = self.f0_params)
            self.vuv_train_function = self.vuv_optimizer.minimize(self.vuv_loss, global_step = self.global_step_vuv, var_list = self.vuv_params)

    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """

        self.harm_loss = tf.reduce_sum(tf.abs(self.harm - self.harm_placeholder)*np.linspace(1.0,0.7,60))

        self.ap_loss = tf.reduce_sum(tf.abs(self.ap - self.ap_placeholder))

        self.vuv_loss = tf.reduce_mean(tf.reduce_mean(binary_cross(self.vuv_placeholder, self.vuv)))

        if config.f0_mode == "cont":

            self.f0_loss = tf.reduce_sum(tf.abs(self.f0 - self.f0_placeholder)*(1-self.vuv_placeholder)) 
        elif config.f0_mode == "discrete":
            self.f0_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.f0_placeholder, logits = self.f0))

        self.loss = self.harm_loss + self.ap_loss 



    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """

        self.harm_summary = tf.summary.scalar('harm_loss', self.harm_loss)

        self.ap_summary = tf.summary.scalar('ap_loss', self.ap_loss)

        self.f0_summary = tf.summary.scalar('f0_loss', self.f0_loss)

        self.vuv_summary = tf.summary.scalar('vuv_loss', self.vuv_loss)

        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()

    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.stft_features),name='input_placeholder')
        self.harm_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,60),name='harm_placeholder')
        self.ap_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,4),name='ap_placeholder')
        if config.f0_mode == "cont":
            self.f0_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='f0_placeholder')
        elif config.f0_mode == "discrete":
            self.f0_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.cqt_bins),name='f0_placeholder')
        self.vuv_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='vuv_placeholder')
        self.is_train = tf.placeholder(tf.bool, name="is_train")

    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()

        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir)
        self.get_summary(sess, config.log_dir)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):
            data_generator = data_gen()
            val_generator = data_gen(mode = 'Val')
            start_time = time.time()


            batch_num = 0
            epoch_final_loss = 0
            epoch_harm_loss = 0
            epoch_ap_loss = 0
            epoch_vuv_loss = 0
            epoch_f0_loss = 0

            val_final_loss = 0
            val_harm_loss = 0
            val_ap_loss = 0
            val_vuv_loss = 0
            val_f0_loss = 0

            with tf.variable_scope('Training'):

                for voc, feat, atb in data_generator:

                    harm_loss, ap_loss, f0_loss, vuv_loss, summary_str = self.train_model(voc, feat, atb, sess)

                    epoch_harm_loss+=harm_loss
                    epoch_ap_loss+=ap_loss
                    epoch_f0_loss+=f0_loss
                    epoch_vuv_loss+=vuv_loss

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_harm_loss = epoch_harm_loss/batch_num
                epoch_ap_loss = epoch_ap_loss/batch_num
                epoch_f0_loss = epoch_f0_loss/batch_num
                epoch_vuv_loss = epoch_vuv_loss/batch_num

                print_dict = {"Harm Loss": epoch_harm_loss}
                print_dict["Ap Loss"] =  epoch_ap_loss
                print_dict["F0 Loss"] =  epoch_f0_loss
                print_dict["Vuv Loss"] =  epoch_vuv_loss

            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for voc, feat, atb in val_generator:

                        harm_loss, ap_loss, f0_loss, vuv_loss, summary_str = self.validate_model(voc, feat, atb, sess)
                        val_harm_loss+=harm_loss
                        val_ap_loss+=ap_loss
                        val_f0_loss+=f0_loss
                        val_vuv_loss+=vuv_loss

                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_harm_loss = val_harm_loss/batch_num
                    val_ap_loss = val_ap_loss/batch_num
                    val_f0_loss = val_f0_loss/batch_num
                    val_vuv_loss = val_vuv_loss/batch_num

                    print_dict["Val Harm Loss"] =  val_harm_loss
                    print_dict["Val Ap Loss"] =  val_ap_loss
                    print_dict["Val F0 Loss"] =  val_f0_loss
                    print_dict["Val Vuv Loss"] =  val_vuv_loss

            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)

    def train_model(self, voc, feat, atb, sess):
        """
        Function to train the model for each epoch
        """

        if config.f0_mode == "cont":
            feed_dict = {self.input_placeholder: voc, self.harm_placeholder: feat[:,:,:60], self.ap_placeholder: feat[:,:,60:64], \
            self.f0_placeholder: feat[:,:,-2:-1], self.vuv_placeholder: feat[:,:,-1:], self.is_train: True}
        elif config.f0_mode == "discrete":
            feed_dict = {self.input_placeholder: voc, self.harm_placeholder: feat[:,:,:60], self.ap_placeholder: feat[:,:,60:64], \
            self.f0_placeholder: atb, self.vuv_placeholder: feat[:,:,-1:], self.is_train: True}

        _,_,_, harm_loss, ap_loss, f0_loss, vuv_loss = sess.run([self.harm_train_function,self.f0_train_function, self.vuv_train_function,
            self.harm_loss, self.ap_loss, self.f0_loss, self.vuv_loss ], feed_dict=feed_dict)



        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return harm_loss, ap_loss, f0_loss, vuv_loss, summary_str

    def validate_model(self, voc, feat, atb, sess):
        """
        Function to train the model for each epoch
        """
        if config.f0_mode == "cont":
            feed_dict = {self.input_placeholder: voc, self.harm_placeholder: feat[:,:,:60], self.ap_placeholder: feat[:,:,60:64], \
            self.f0_placeholder: feat[:,:,-2:-1], self.vuv_placeholder: feat[:,:,-1:], self.is_train: False}
        elif config.f0_mode == "discrete":
            feed_dict = {self.input_placeholder: voc, self.harm_placeholder: feat[:,:,:60], self.ap_placeholder: feat[:,:,60:64], \
            self.f0_placeholder: atb, self.vuv_placeholder: feat[:,:,-1:], self.is_train: False}
        harm_loss, ap_loss, f0_loss, vuv_loss = sess.run([self.harm_loss, self.ap_loss, self.f0_loss, self.vuv_loss ], feed_dict=feed_dict)


        summary_str = sess.run(self.summary, feed_dict=feed_dict)

        return harm_loss, ap_loss, f0_loss, vuv_loss, summary_str



    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """

        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')
        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])

        stat_file.close()

        with h5py.File(config.voice_dir + file_name) as voc_file:

            voc_stft = voc_file["voc_stft"][()]

            back_stft = voc_file["back_stft"][()]

            mix_stft = (voc_stft + back_stft)/2

            feats = np.array(voc_file['world_feats'])

            atb = voc_file["atb"][()]

            atb = atb[:, 1:]


        return mix_stft, feats, atb

    def test_file_hdf5(self, file_name):
        """
        Function to extract vocals from hdf5 file.
        """
        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        mix_stft, feats, atb = self.read_hdf5_file(file_name)
        out_feats, out_atb, out_vuv = self.process_file(mix_stft,  sess)

        if config.f0_mode == "discrete":

            est_freq = utils.to_viterbi_cents(out_atb)

            est_freq = est_freq/100
            est_freq = est_freq + 12*np.log2(10) - 12*np.log2(440)
            est_freq = est_freq + 69 

            ori_freq = utils.to_viterbi_cents(atb)

            ori_freq = ori_freq/100
            ori_freq = ori_freq + 12*np.log2(10) - 12*np.log2(440)
            ori_freq = ori_freq + 69

            # import pdb;pdb.set_trace()

        elif config.f0_mode == "cont":
            est_freq = None

        self.plot_features(feats, out_feats, mix_stft, atb, out_atb, est_freq, out_vuv)

        # import pdb;pdb.set_trace()

        synth = utils.query_yes_no("Synthesize output? ")

        if synth:

            if config.f0_mode == "cont":

                audio_out = utils.feats_to_audio(np.concatenate((out_feats[:feats.shape[0],:-1], out_vuv[:feats.shape[0]]) , axis = -1))
                sf.write('./{}_ss_pred.wav'.format(file_name[:-5]), audio_out, config.fs)

            elif config.f0_mode == "discrete":

                audio_out = utils.feats_to_audio(np.concatenate((out_feats[:feats.shape[0]], np.expand_dims(est_freq[:feats.shape[0]],-1), out_vuv[:feats.shape[0]]) , axis = -1))
                sf.write('./{}_ss_pred_dis.wav'.format(file_name[:-5]), audio_out, config.fs)


            # sf.write('./{}_ss_pred.wav'.format(file_name[:-5]), audio_out, config.fs)

        synth = utils.query_yes_no("Synthesize output with original F0? ")

        if synth:

            audio_out = utils.feats_to_audio(np.concatenate((out_feats[:feats.shape[0],:-2], feats[:out_feats.shape[0],-2:]) , axis = -1))

            sf.write('./{}_ss_ori.wav'.format(file_name[:-5]), audio_out, config.fs)

        synth_ori = utils.query_yes_no("Synthesize ground truth with vocoder? ")

        if synth_ori:
            audio = utils.feats_to_audio(feats) 
            sf.write('./{}_ori.wav'.format(file_name[:-5]), audio, config.fs)

    def read_wav_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs)

        audio = np.float64(audio)

        if len(audio.shape) == 2:

            vocals = np.array((audio[:,1]+audio[:,0])/2)

        else: 
            vocals = np.array(audio)

        voc_stft = np.clip(abs(utils.stft(vocals, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)), 0.0, 1.0)

        return voc_stft

    def read_acap_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs)

        audio = np.float64(audio)

        if len(audio.shape) == 2:

            vocals = np.array((audio[:,1]+audio[:,0])/2)

        else: 
            vocals = np.array(audio)

        feats, f0 = utils.stft_to_feats(vocals)

        return feats

    def test_file_wav(self, file_name, acap_file=None):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir =  config.log_dir)
        mel = self.read_wav_file(file_name)

        if acap_file:
            feats = self.read_acap_file(acap_file)
        else:
            feats = None


        out_mel, out_atb, out_vuv = self.process_file(mel, sess)
        if config.f0_mode == "discrete":

            est_freq = utils.to_viterbi_cents(out_atb)

            est_freq = est_freq/100
            est_freq = est_freq + 12*np.log2(10) - 12*np.log2(440)
            est_freq = est_freq + 69 

        plt.figure(1)

        if acap_file:

            ax1 = plt.subplot(311)

            plt.imshow(np.log(mel.T),aspect='auto',origin='lower')

            ax1.set_title("Input STFT", fontsize=10)

            ax2 = plt.subplot(312, sharex = ax1)

            plt.imshow(feats[:,:64].T,aspect='auto',origin='lower')

            ax2.set_title("Ground Truth Vocoder Features", fontsize=10)

            ax3 = plt.subplot(313, sharex = ax1, sharey = ax2)

            plt.imshow(out_mel[:feats.shape[0]].T,aspect='auto',origin='lower')

            ax3.set_title("Output Vocoder Features", fontsize=10)


            plt.figure(4)


            ax1 = plt.subplot(211)

            plt.plot(feats[:,-1])

            ax1.set_title("Ground Truth VUV", fontsize=10)

            ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)

            plt.plot(out_vuv)

            ax1.set_title("Output VUV", fontsize=10)


            plt.figure(3)



            if config.f0_mode == "cont":
                f0_output = out_mel[:feats.shape[0],-1]
            else:
                f0_output = est_freq

            f0_output = f0_output*(1-out_vuv[:feats.shape[0],0])
            f0_output[f0_output == 0] = np.nan


            plt.plot(f0_output, label = "Predicted Value")
            f0_gt = feats[:,-2]
            f0_gt = f0_gt*(1-feats[:,-1])
            f0_gt[f0_gt == 0] = np.nan
            plt.plot(f0_gt, label="Ground Truth")
            f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
            f0_greater = np.where(f0_difference>config.f0_threshold)
            diff_per = f0_greater[0].shape[0]/len(f0_output)
            plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))
            plt.legend()

            plt.show()

        else:


            ax1 = plt.subplot(211)

            plt.imshow(np.log(mel.T),aspect='auto',origin='lower')

            ax1.set_title("Input STFT", fontsize=10)

            ax1 = plt.subplot(212)

            plt.imshow(out_mel.T,aspect='auto',origin='lower')

            ax1.set_title("Output Vocoder Features", fontsize=10)

            plt.show()



        if config.f0_mode == "cont":

            audio_out = utils.feats_to_audio(np.concatenate((out_mel, out_vuv) , axis = -1))
            sf.write('./{}_ss_pred.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)

        elif config.f0_mode == "discrete":

            audio_out = utils.feats_to_audio(np.concatenate((out_mel, np.expand_dims(est_freq,-1), out_vuv) , axis = -1))
            sf.write('./{}_ss_pred_dis.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)

        if acap_file:

            audio = utils.feats_to_audio(feats) 
            sf.write('./{}_ori.wav'.format(file_name.split('/')[-1][:-4]), audio, config.fs)

        np.save(file_name.split('/')[-1][:-4], out_mel)



    def plot_features(self, feats, out_feats, mix_stft, atb, out_atb, est_frequ, out_vuv):
        """
        Function to plot output and ground truth features
        """
        plt.figure(1)


        ax1 = plt.subplot(311)

        plt.imshow(np.log(mix_stft.T),aspect='auto',origin='lower')

        ax1.set_title("Input STFT", fontsize=10)

        ax2 = plt.subplot(312, sharex = ax1, sharey = ax1)

        plt.imshow(feats[:,:60].T,aspect='auto',origin='lower')

        ax1.set_title("Ground Truth Vocoder Features", fontsize=10)

        ax3 = plt.subplot(313, sharex = ax1, sharey = ax1)

        ax3.set_title("Output Vocoder Features", fontsize=10)

        plt.imshow(out_feats[:,:60].T,aspect='auto',origin='lower')

        plt.figure(4)


        ax1 = plt.subplot(211)

        plt.plot(feats[:,-1])

        ax1.set_title("Ground Truth VUV", fontsize=10)

        ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)

        plt.plot(out_vuv)

        ax1.set_title("Output VUV", fontsize=10)

        if config.f0_mode == "cont":

            plt.figure(2)
            f0_output = out_feats[:feats.shape[0],-2]
            f0_output = f0_output*(1-feats[:,-1])
            f0_output[f0_output == 0] = np.nan
            plt.plot(f0_output, label = "Predicted Value")
            f0_gt = feats[:,-2]
            f0_gt = f0_gt*(1-feats[:,-1])
            f0_gt[f0_gt == 0] = np.nan
            plt.plot(f0_gt, label="Ground Truth")
            f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
            f0_greater = np.where(f0_difference>config.f0_threshold)
            diff_per = f0_greater[0].shape[0]/len(f0_output)
            plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))

        elif config.f0_mode == "discrete":
            out_atb = out_atb[:atb.shape[0]]
            time_1, ori_freq = utils.process_output(atb)
            time_2, est_freq = utils.process_output(out_atb)
            # import pdb;pdb.set_trace()
            scores = mir_eval.multipitch.evaluate(time_1, ori_freq, time_2, est_freq)
            pre = scores['Precision']
            acc = scores['Accuracy']
            rec = scores['Recall']
            plt.figure(2)
            ax1 = plt.subplot(211)
            plt.imshow(atb.T, origin='lower', aspect='auto')
            ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)
            plt.imshow(out_atb.T, origin='lower', aspect='auto')
            plt.suptitle("Precision: {:.3%}, Accuracy: {:.3%},  Recall: {:.3%}".format(pre, acc, rec))

            plt.figure(3)
            f0_output = est_frequ[:feats.shape[0]]
            f0_output = f0_output*(1-feats[:,-1])
            f0_output[f0_output == 0] = np.nan


            plt.plot(f0_output, label = "Predicted Value")
            f0_gt = feats[:,-2]
            f0_gt = f0_gt*(1-feats[:,-1])
            f0_gt[f0_gt == 0] = np.nan
            plt.plot(f0_gt, label="Ground Truth")
            f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
            f0_greater = np.where(f0_difference>config.f0_threshold)
            diff_per = f0_greater[0].shape[0]/len(f0_output)
            plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))



        plt.show()


    def process_file(self, mix_stft, sess):

        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        in_batches_stft, nchunks_in = utils.generate_overlapadd(mix_stft)

        out_batches_feats = []
        out_vuv = []
        if config.f0_mode == "discrete":
            out_atb = []

        for in_batch_stft in in_batches_stft :
            feed_dict = {self.input_placeholder: in_batch_stft, self.is_train: False}
            harm, ap, f0, vuv = sess.run([self.harm, self.ap, self.f0, self.vuv], feed_dict=feed_dict)
            out_vuv.append(vuv)
            if config.f0_mode == "cont":
                val_feats = np.concatenate((harm, ap, f0), axis=-1)
                out_batches_feats.append(val_feats)
            elif config.f0_mode == "discrete":
                val_feats = np.concatenate((harm, ap), axis=-1)
                out_batches_feats.append(val_feats)
                out_atb.append(f0)

        out_batches_feats = np.array(out_batches_feats)

        out_feats = utils.overlapadd(out_batches_feats,nchunks_in)

        out_vuv = utils.overlapadd(np.array(out_vuv),nchunks_in)

        if config.f0_mode == "discrete":

            out_atb = utils.overlapadd(np.array(out_atb), nchunks_in)

        if config.f0_mode == "cont":

            out_feats = out_feats*(max_feat[:-1]-min_feat[:-1])+min_feat[:-1]
            out_atb = None

        elif config.f0_mode == "discrete":
            out_feats = out_feats*(max_feat[:-2]-min_feat[:-2])+min_feat[:-2]

        out_vuv = np.round(out_vuv)


        return out_feats, out_atb, out_vuv

    def read_med_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs, mono=False)

        audio = np.float64(audio)

        vocals = np.array(audio[1,:])

        mixture = np.array(audio[0,:])

        voc_stft = np.clip(abs(utils.stft(mixture, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)), 0.0, 1.0)

        feats, f0 = utils.stft_to_feats(vocals)

        return voc_stft, feats, mixture

    def test_folder_wav(self, folder_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir=config.log_dir)

        file_list = [x for x in os.listdir(folder_name) if x in config.med_to_use]

        count = 0

        unprocessable = []

        for file_name in file_list:
            # try:
            mel, feats, mixture = self.read_med_file(os.path.join(folder_name, file_name))
            out_mel, out_f0, out_vuv = self.process_file(mel, sess)
            out_featss = np.concatenate((out_mel, out_vuv), axis = -1)

            audio_out = utils.feats_to_audio(out_featss) 

            audio_ori = utils.feats_to_audio(feats) 

            sf.write(os.path.join(config.output_dir,'./{}_output_1.wav'.format(file_name.split('/')[-1][:-4])), audio_out, config.fs)

            # except:
            #     unprocessable.append(file_name)

            count+=1

            utils.progress(count, len(file_list), "Files processed")

        print(unprocessable)

    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """
        with tf.variable_scope('Harm_Model') as scope:
            if config.enc_mode == "wave":
                self.harm, self.ap = modules.nr_wavenet(self.input_placeholder, self.is_train)
            elif config.enc_mode == "conv":
                self.harm, self.ap = modules.enc_dec(self.input_placeholder, self.is_train)
        with tf.variable_scope('F0_Model') as scope:
            if config.enc_mode == "wave":
                self.f0 = modules.nr_wavenet_f0(self.input_placeholder, self.harm, self.ap, self.is_train)
            elif config.enc_mode == "conv":
                self.f0 = modules.enc_dec_f0(self.input_placeholder, self.harm, self.ap, self.is_train)
        with tf.variable_scope('Vuv_Model') as scope:
            if config.enc_mode == "wave":
                self.vuv = modules.nr_wavenet_vuv(self.input_placeholder, self.harm, self.ap, self.f0, self.is_train)
            elif config.enc_mode == "conv":
                self.vuv = modules.enc_dec_vuv(self.input_placeholder, self.harm, self.ap, self.f0, self.is_train)


class SSSynth_Content(Model):

    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        auto_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='post_net')

        self.auto_saver = tf.train.Saver(max_to_keep= config.max_models_to_keep, var_list = auto_var_list)

        # if not self.stft_var_list:
        self.stft_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_encoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_post_net') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'F0_Model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Vuv_Model')

        self.stft_saver = tf.train.Saver(max_to_keep= config.max_models_to_keep, var_list = self.stft_var_list)

        sess.run(self.init_op)

        ckpt_auto = tf.train.get_checkpoint_state(config.log_dir)

        if ckpt_auto and ckpt_auto.model_checkpoint_path:
            print("Using the AUTOVC model in %s"%ckpt_auto.model_checkpoint_path)
            self.auto_saver.restore(sess, ckpt_auto.model_checkpoint_path)

        ckpt_stft = tf.train.get_checkpoint_state(config.log_dir_2)

        if ckpt_stft and ckpt_stft.model_checkpoint_path:
            print("Using the STFT model in %s"%ckpt_stft.model_checkpoint_path)
            self.stft_saver.restore(sess, ckpt_stft.model_checkpoint_path)

    def save_model(self, sess, epoch, log_dir):
        """
        Save the model.
        """
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        self.stft_saver.save(sess, checkpoint_file, global_step=epoch)
    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """

        self.optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_f0 = tf.Variable(0, name='global_step_f0', trainable=False)
        self.global_step_vuv = tf.Variable(0, name='global_step_vuv', trainable=False)

        self.harm_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_encoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_post_net')
        self.f0_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'F0_Model')
        self.vuv_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Vuv_Model')

        self.f0_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)
        self.vuv_optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.optimizer.minimize(self.final_loss, global_step=self.global_step, var_list=self.harm_params)
            self.f0_train_function = self.f0_optimizer.minimize(self.f0_loss, global_step=self.global_step_f0, var_list=self.f0_params)
            self.vuv_train_function = self.vuv_optimizer.minimize(self.vuv_loss, global_step=self.global_step_vuv, var_list=self.vuv_params)


    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """

        self.recon_loss = tf.reduce_sum(tf.square(self.input_placeholder - self.output_stft) ) 

        self.content_loss = tf.reduce_sum(tf.abs(self.content_embedding_1 - self.content_embedding_stft))

        self.content_loss_2 = tf.reduce_sum(tf.abs(self.content_embedding_1 - self.content_embedding_stft_2))

        self.content_loss = self.content_loss + self.content_loss_2

        self.recon_loss_0 = tf.reduce_sum(tf.square(self.input_placeholder - self.output_stft_1))

        self.final_loss = self.recon_loss + config.mu * self.recon_loss_0 + config.lamda * self.content_loss

        self.vuv_loss = tf.reduce_mean(tf.reduce_mean(binary_cross(self.vuv_placeholder, self.vuv)))

        if config.f0_mode == "cont":

            self.f0_loss = tf.reduce_sum(tf.abs(self.f0 - self.f0_placeholder)*(1-self.vuv_placeholder)) 
        elif config.f0_mode == "discrete":
            self.f0_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.f0_placeholder, logits = self.f0))


    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """


        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)

        self.recon_summary = tf.summary.scalar('recon_loss', self.recon_loss)

        self.recon_0_summary = tf.summary.scalar('recon_0_loss', self.recon_loss_0)

        self.content_summary = tf.summary.scalar('content_loss', self.content_loss)

        self.f0_summary = tf.summary.scalar('f0_loss', self.f0_loss)

        self.vuv_summary = tf.summary.scalar('vuv_loss', self.vuv_loss)

        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()


    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.input_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.output_features),
                                           name='input_placeholder')       


        self.stft_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.stft_features),
                                           name='stft_placeholder')  

        self.speaker_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        self.speaker_onehot_labels = tf.one_hot(indices=tf.cast(self.speaker_labels, tf.int32), depth = config.num_speakers)

        self.speaker_labels_1 = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        self.speaker_onehot_labels_1 = tf.one_hot(indices=tf.cast(self.speaker_labels_1, tf.int32), depth = config.num_speakers)

        if config.f0_mode == "cont":
            self.f0_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='f0_placeholder')
        elif config.f0_mode == "discrete":
            self.f0_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,config.cqt_bins),name='f0_placeholder')
        self.vuv_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size,config.max_phr_len,1),name='vuv_placeholder')

        self.is_train = tf.placeholder(tf.bool, name="is_train")


    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()

        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir)

        self.get_summary(sess, config.log_dir_2)

        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):

            data_generator = data_gen_stft()
            val_generator = data_gen_stft(mode = 'Val')
            

            epoch_final_loss = 0
            epoch_recon_loss = 0
            epoch_recon_0_loss = 0
            epoch_content_loss = 0

            val_final_loss = 0
            val_recon_loss = 0
            val_recon_0_loss = 0
            val_content_loss = 0

            batch_num = 0

            start_time = time.time()

            with tf.variable_scope('Training'):
                for feats_targs, stft_targs, targets_speakers in data_generator:
                    final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss, summary_str = self.train_model(feats_targs, stft_targs, targets_speakers, sess)
                    epoch_final_loss+=final_loss
                    epoch_recon_loss+=recon_loss
                    epoch_recon_0_loss+=recon_loss_0
                    epoch_content_loss+=content_loss

                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num
                epoch_recon_loss = epoch_recon_loss/batch_num
                epoch_recon_0_loss = epoch_recon_0_loss/batch_num
                epoch_content_loss = epoch_content_loss/batch_num

                print_dict = {"Final Loss": epoch_final_loss}

                print_dict["Recon Loss"] =  epoch_recon_loss
                print_dict["Recon Loss_0 "] =  epoch_recon_0_loss
                print_dict["Content Loss"] =  epoch_content_loss



            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for feats_targs, stft_targs, targets_speakers in val_generator:


                        final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss, summary_str = self.validate_model(feats_targs, stft_targs, targets_speakers, sess)

                        val_final_loss+=final_loss
                        val_recon_loss+=recon_loss
                        val_recon_0_loss+=recon_loss_0
                        val_content_loss+=content_loss

                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()

                        utils.progress(batch_num,config.batches_per_epoch_val, suffix = 'validation done')

                        batch_num+=1

                    val_final_loss = val_final_loss/batch_num
                    val_recon_loss = val_recon_loss/batch_num
                    val_recon_0_loss = val_recon_0_loss/batch_num
                    val_content_loss = val_content_loss/batch_num

                    print_dict["Val Final Loss"] = val_final_loss

                    print_dict["Val Recon Loss"] =  val_recon_loss
                    print_dict["Val Recon Loss_0 "] =  val_recon_0_loss
                    print_dict["Val Content Loss"] =  val_content_loss



            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir_2)


    def train_model(self,feats_targs, stft_targs, targets_speakers, sess):
        """
        Function to train the model for each epoch
        """

        if config.use_f0_emb:
            feed_dict = {self.input_placeholder: feats_targs, self.stft_placeholder: stft_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers, self.f0_placeholder: feats_targs[:,:,-2:-1], self.vuv_placeholder: feats_targs[:,:,-1:],self.is_train: True}
        else:

            feed_dict = {self.input_placeholder: feats_targs[:,:,:64], self.stft_placeholder: stft_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers, self.f0_placeholder: feats_targs[:,:,-2:-1], self.vuv_placeholder: feats_targs[:,:,-1:],self.is_train: True}
            
        _,_,_, final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss = sess.run([self.final_train_function, self.f0_train_function, self.vuv_train_function, self.final_loss, self.recon_loss, self.recon_loss_0, self.content_loss, self.f0_loss, self.vuv_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss, summary_str
 

    def validate_model(self,feats_targs, stft_targs, targets_speakers, sess):
        """
        Function to train the model for each epoch
        """


        if config.use_f0_emb:
            feed_dict = {self.input_placeholder: feats_targs, self.stft_placeholder: stft_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers, self.f0_placeholder: feats_targs[:,:,-2:-1], self.vuv_placeholder: feats_targs[:,:,-1:],self.is_train: False}
        else:

            feed_dict = {self.input_placeholder: feats_targs[:,:,:64], self.stft_placeholder: stft_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers, self.f0_placeholder: feats_targs[:,:,-2:-1], self.vuv_placeholder: feats_targs[:,:,-1:],self.is_train: False}
            
        final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss = sess.run([self.final_loss, self.recon_loss, self.recon_loss_0, self.content_loss, self.f0_loss, self.vuv_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss, summary_str



    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        # if file_name.endswith('.hdf5'):

        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = stat_file["feats_maximus"][()]
        min_feat = stat_file["feats_minimus"][()]


        stat_file.close()  

        with h5py.File(config.voice_dir+file_name, "r") as hdf5_file:
            mel = hdf5_file["world_feats"][()]

            if config.mix_aug:

                voc_stft = hdf5_file["voc_stft"][()]

                back_stft = hdf5_file["back_stft"][()]

                stft = (voc_stft + back_stft)/2 

            else:

                stft = hdf5_file["voc_stft"][()]        

        return mel, stft

    def read_wav_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs)

        audio = np.float64(audio)

        if len(audio.shape) == 2:

            vocals = np.array((audio[:,1]+audio[:,0])/2)

        else: 
            vocals = np.array(audio)

        voc_stft = np.clip(abs(utils.stft(vocals, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)), 0.0, 1.0)

        return voc_stft

    def read_acap_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs)

        audio = np.float64(audio)

        if len(audio.shape) == 2:

            vocals = np.array((audio[:,1]+audio[:,0])/2)

        else: 
            vocals = np.array(audio)

        feats, f0 = utils.stft_to_feats(vocals)

        return feats

    def test_file_wav(self, file_name, acap_file=None):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir =  config.log_dir)
        mel = self.read_wav_file(file_name)

        if acap_file:
            feats = self.read_acap_file(acap_file)
        else:
            feats = None


        out_mel, out_f0, out_vuv = self.process_file(mel, sess)

        plt.figure(1)

        if acap_file:

            ax1 = plt.subplot(311)

            plt.imshow(np.log(mel.T),aspect='auto',origin='lower')

            ax1.set_title("Input STFT", fontsize=10)

            ax2 = plt.subplot(312, sharex = ax1)

            plt.imshow(feats[:,:64].T,aspect='auto',origin='lower')

            ax2.set_title("Ground Truth Vocoder Features", fontsize=10)

            ax3 = plt.subplot(313, sharex = ax1, sharey = ax2)

            plt.imshow(out_mel[:feats.shape[0]].T,aspect='auto',origin='lower')

            ax3.set_title("Output Vocoder Features", fontsize=10)


            plt.figure(4)


            ax1 = plt.subplot(211)

            plt.plot(feats[:,-1])

            ax1.set_title("Ground Truth VUV", fontsize=10)

            ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)

            plt.plot(out_vuv)

            ax1.set_title("Output VUV", fontsize=10)


            plt.figure(3)
            f0_output = out_f0[:feats.shape[0],0]

            f0_output = f0_output*(1-out_vuv[:feats.shape[0],0])
            f0_output[f0_output == 0] = np.nan


            plt.plot(f0_output, label = "Predicted Value")
            f0_gt = feats[:,-2]
            f0_gt = f0_gt*(1-feats[:,-1])
            f0_gt[f0_gt == 0] = np.nan
            plt.plot(f0_gt, label="Ground Truth")
            f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
            f0_greater = np.where(f0_difference>config.f0_threshold)
            diff_per = f0_greater[0].shape[0]/len(f0_output)
            plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))
            plt.legend()

            plt.show()

        else:


            ax1 = plt.subplot(211)

            plt.imshow(np.log(mel.T),aspect='auto',origin='lower')

            ax1.set_title("Input STFT", fontsize=10)

            ax1 = plt.subplot(212)

            plt.imshow(out_mel.T,aspect='auto',origin='lower')

            ax1.set_title("Output Vocoder Features", fontsize=10)

            plt.show()

        out_featss = np.concatenate((out_mel, out_f0, out_vuv), axis = -1)

        audio_out = utils.feats_to_audio(out_featss) 

        sf.write('./{}_output.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)

        if acap_file:

            audio = utils.feats_to_audio(feats) 
            sf.write('./{}_ori.wav'.format(file_name.split('/')[-1][:-4]), audio, config.fs)

        np.save(file_name.split('/')[-1][:-4], out_mel)


    def test_file_wav_f0(self, file_name, f0_file):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir =  config.log_dir)

        mel = self.read_wav_file(file_name)

        f0 = midi_process.open_f0_file(f0_file)

        timestamps = np.arange(0, len(mel)*config.hoptime, config.hoptime)


        f1 = vamp_notes.note2traj(f0, timestamps)

        f1 = sig_process.process_pitch(f1[:,0])

        out_mel, out_f0, out_vuv = self.process_file(mel, sess)

        plot_dict = {"Spec Envelope": {"gt": mel[:,:-6], "op": out_mel[:,:-4]}, "Aperiodic":{"gt": mel[:,-6:-2], "op": out_mel[:,-4:]},\
         "F0": {"gt": f1[:,0], "op": out_f0}, "Vuv": {"gt": f1[:,1], "op": out_vuv}}


        self.plot_features(plot_dict)

        synth = utils.query_yes_no("Synthesize output? ")

        file_name = file_name.split('/')[-1]

        if synth:

            out_featss = np.concatenate((out_mel[:f1.shape[0]], f1[:,0:1], out_vuv[:f1.shape[0]]), axis = -1)

            audio_out = utils.feats_to_audio(out_featss) 

            sf.write(os.path.join(config.output_dir,'{}_SIN_YAM_f0_{}.wav'.format(file_name[:-4], f0_file.split('/')[-1])), audio_out, config.fs)

        synth_ori = utils.query_yes_no("Synthesize with output f0? ")

        if synth_ori:
            out_featss = np.concatenate((out_mel, out_f0, out_vuv), axis = -1)

            audio_out = utils.feats_to_audio(out_featss) 

            sf.write('./{}_SIN_YAM_OutF0.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)

    def test_file_hdf5(self, file_name, speaker_index=0, speaker_index_2=0):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir =  config.log_dir)
        mel, stft = self.read_hdf5_file(file_name)



        out_mel, out_f0, out_vuv = self.process_file(stft, sess)

        stft, mel, out_mel, out_f0, out_vuv = utils.match_time([stft, mel, out_mel, out_f0, out_vuv])


        self.plot_features(stft, mel, out_mel, out_f0, out_vuv)



        synth = utils.query_yes_no("Synthesize output? ")

        if synth:
            gen_change = utils.query_yes_no("Change in gender? ")
            if gen_change:
                female_male = utils.query_yes_no("Female to male?")
                if female_male:
                    out_featss = np.concatenate((out_mel[:mel.shape[0]], mel[:out_mel.shape[0],-2:-1]-12, mel[:out_mel.shape[0],-1:]), axis = -1)
                else:
                    out_featss = np.concatenate((out_mel[:mel.shape[0]], mel[:out_mel.shape[0],-2:-1]+12, mel[:out_mel.shape[0],-1:]), axis = -1)
            else:
                out_featss = np.concatenate((out_mel[:mel.shape[0]], mel[:out_mel.shape[0],-2:]), axis = -1)

            audio_out = utils.feats_to_audio(out_featss) 

            sf.write('./{}_{}_STFT.wav'.format(file_name[:-5], config.singers[speaker_index_2]), audio_out, config.fs)

        synth_ori = utils.query_yes_no("Synthesize ground truth with vocoder? ")

        if synth_ori:
            audio = utils.feats_to_audio(mel) 
            sf.write('./{}_{}_ori.wav'.format(file_name[:-5], config.singers[speaker_index]), audio, config.fs)

    def plot_features(self, feat_dict):
        """
        Plots a set of features, with the ground truth and the output as in the directory.
        """

        for num, feature in enumerate(feat_dict.keys()):
            plt.figure(num)
            gt = feat_dict[feature]['gt']
            op = feat_dict[feature]['op']
            if len(gt.shape) == 1 or gt.shape[-1] == 1:
                plt.plot(gt, label = "Ground Truth {}".format(feature))
                plt.plot(op, label = "Output {}".format(feature))
                if "notes" in feat_dict[feature].keys():
                    plt.plot(feat_dict[feature]["notes"], label = "Notes")
                plt.legend()
            else:
                ax1 = plt.subplot(211)

                plt.imshow(gt.T,aspect='auto',origin='lower')

                ax1.set_title("Ground Truth {}".format(feature, fontsize=10))

                ax2 =plt.subplot(212, sharex = ax1, sharey = ax1)

                ax2.set_title("Output {}".format(feature, fontsize=10))

                plt.imshow(op.T,aspect='auto',origin='lower')

                ax2.set_title("Output {}".format(feature, fontsize=10))

        
        plt.show()
    # def plot_features(self, stft, feats, out_feats, out_f0, out_vuv):

    #     plt.figure(1)


    #     ax1 = plt.subplot(311)

    #     plt.imshow(np.log(stft.T),aspect='auto',origin='lower')

    #     ax1.set_title("Input STFT", fontsize=10)

    #     ax1 = plt.subplot(312)

    #     plt.imshow(feats[:,:64].T,aspect='auto',origin='lower')

    #     ax1.set_title("Ground Truth Vocoder Features", fontsize=10)

    #     ax3 = plt.subplot(313, sharex = ax1, sharey = ax1)

    #     ax3.set_title("Output Vocoder Features", fontsize=10)

    #     plt.imshow(out_feats[:,:64].T,aspect='auto',origin='lower')


    #     plt.figure(4)

    #     ax1 = plt.subplot(211)

    #     plt.plot(feats[:,-1])

    #     ax1.set_title("Ground Truth VUV", fontsize=10)

    #     ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)

    #     plt.plot(out_vuv)

    #     ax1.set_title("Output VUV", fontsize=10)

    #     if config.f0_mode == "cont":

    #         plt.figure(2)
    #         f0_output = np.squeeze(out_f0)
    #         f0_output = f0_output*(1-feats[:,-1])
    #         f0_output[f0_output == 0] = np.nan
    #         # import pdb;pdb.set_trace()
    #         plt.plot(f0_output, label = "Predicted Value")
    #         f0_gt = feats[:,-2]
    #         f0_gt = f0_gt*(1-feats[:,-1])
    #         f0_gt[f0_gt == 0] = np.nan
    #         plt.plot(f0_gt, label="Ground Truth")
    #         f0_difference = np.nan_to_num(abs(f0_gt-f0_output))
    #         f0_greater = np.where(f0_difference>config.f0_threshold)
    #         diff_per = f0_greater[0].shape[0]/len(f0_output)
    #         plt.suptitle("Percentage correct = "+'{:.3%}'.format(1-diff_per))
    #         plt.legend()
    #     plt.show()


    def process_file(self, mel, sess):

        if config.use_casas:

            stat_file = h5py.File('./stats.hdf5', mode='r')
        else:
            stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = stat_file["feats_maximus"][()]
        min_feat = stat_file["feats_minimus"][()]


        stat_file.close()  

        mel = np.clip(mel, 0.0, 1.0)


        in_batches_mel, nchunks_in = utils.generate_overlapadd(mel)

        out_batches_mel = []
        out_f0 = []
        out_vuv = []

        for in_batch_mel in in_batches_mel :
            # speaker = np.repeat(speaker_index, config.batch_size)
            # speaker_2 = np.repeat(speaker_index_2, config.batch_size)
            feed_dict = {self.stft_placeholder: in_batch_mel, self.is_train: False}
            mel, f0, vuv = sess.run([self.output_stft, self.f0, self.vuv], feed_dict=feed_dict)
            out_batches_mel.append(mel)
            out_f0.append(f0)
            out_vuv.append(vuv)

        out_batches_mel = np.array(out_batches_mel)

        out_batches_mel = utils.overlapadd(out_batches_mel,nchunks_in)
        out_f0 = utils.overlapadd(np.array(out_f0), nchunks_in)
        out_vuv = utils.overlapadd(np.array(out_vuv), nchunks_in)

        out_f0 = out_f0*(max_feat[-2] - min_feat[-2]) + min_feat[-2]

        if config.use_f0_emb:
            out_batches_mel = out_batches_mel*(max_feat- min_feat) + min_feat
        else:
            out_batches_mel = out_batches_mel*(max_feat[:-2] - min_feat[:-2]) + min_feat[:-2]

        out_vuv = np.round(out_vuv)

        return out_batches_mel, out_f0, out_vuv

    def read_med_file(self, file_name):

        audio, fs = librosa.core.load(file_name, sr=config.fs, mono=False)

        audio = np.float64(audio)

        vocals = np.array(audio[1,:])

        mixture = np.array(audio[0,:])

        voc_stft = np.clip(abs(utils.stft(mixture, hopsize = config.hopsize, nfft = config.nfft, fs = config.fs, window = config.window)), 0.0, 1.0)

        feats, f0 = utils.stft_to_feats(vocals)

        return voc_stft, feats, mixture

    def test_folder_wav(self, folder_name):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir=config.log_dir)

        file_list = [x for x in os.listdir(folder_name) if x in config.med_to_use]

        count = 0

        unprocessable = []

        for file_name in file_list:
            try:
                mel, feats, mixture = self.read_med_file(os.path.join(folder_name, file_name))
                out_mel, out_f0, out_vuv = self.process_file(mel, sess)
                out_featss = np.concatenate((out_mel, out_f0, out_vuv), axis = -1)

                audio_out = utils.feats_to_audio(out_featss) 

                audio_ori = utils.feats_to_audio(feats) 

                sf.write(os.path.join(config.output_dir,'./{}_output.wav'.format(file_name.split('/')[-1][:-4])), audio_out, config.fs)

                sf.write(os.path.join(config.output_dir,'./{}_ori.wav'.format(file_name.split('/')[-1][:-4])), audio_ori, config.fs)

                sf.write(os.path.join(config.output_dir,'./{}_mix.wav'.format(file_name.split('/')[-1][:-4])), mixture, config.fs)

            except:
                unprocessable.append(file_name)

            count+=1

            utils.progress(count, len(file_list), "Files processed")

            print(unprocessable)



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """


        with tf.variable_scope('encoder') as scope:
            self.content_embedding_1 = modules.content_encoder(self.input_placeholder, self.speaker_onehot_labels, self.is_train)

        with tf.variable_scope('decoder') as scope: 
            self.output_1 = modules.decoder(self.content_embedding_1, self.speaker_onehot_labels_1, self.is_train)

        with tf.variable_scope('post_net') as scope: 
            self.residual = modules.post_net(self.output_1, self.is_train)
            self.output = self.output_1 + self.residual
        with tf.variable_scope('encoder') as scope:
            scope.reuse_variables()
            self.content_embedding_2 = modules.content_encoder(self.output, self.speaker_onehot_labels, self.is_train)

        with tf.variable_scope('stft_encoder') as scope:
            self.content_embedding_stft = modules.content_encoder_sep(self.stft_placeholder, self.is_train)

        with tf.variable_scope('stft_decoder') as scope: 
            self.output_stft_1 = modules.decoder_sep(self.content_embedding_stft, self.stft_placeholder, self.is_train)

        with tf.variable_scope('stft_post_net') as scope: 

            self.residual_stft = modules.post_net_sep(self.output_stft_1, self.is_train)

            self.output_stft = self.output_stft_1 + self.residual_stft

        with tf.variable_scope('encoder') as scope:
            scope.reuse_variables()
            self.content_embedding_stft_2 = modules.content_encoder(self.output_stft, self.speaker_onehot_labels, self.is_train)

        with tf.variable_scope('F0_Model') as scope:
            if config.enc_mode == "wave":
                self.f0 = modules.nr_wavenet_f0(self.stft_placeholder, self.output_stft[:,:,:-4], self.output_stft[:,:,-4:], self.is_train)
            elif config.enc_mode == "conv":
                self.f0 = modules.enc_dec_f0(self.stft_placeholder, self.output_stft[:,:,:-4], self.output_stft[:,:,-4:], self.is_train)
        with tf.variable_scope('Vuv_Model') as scope:
            if config.enc_mode == "wave":
                self.vuv = modules.nr_wavenet_vuv(self.stft_placeholder, self.output_stft[:,:,:-4], self.output_stft[:,:,-4:], self.f0, self.is_train)
            elif config.enc_mode == "conv":
                self.vuv = modules.enc_dec_vuv(self.stft_placeholder, self.output_stft[:,:,:-4], self.output_stft[:,:,-4:], self.f0, self.is_train)

def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = MultiSynth()
    model.train()

if __name__ == '__main__':
    test()





