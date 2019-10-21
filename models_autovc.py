import tensorflow as tf
import modules_autovc as modules
import config_autovc as config
from data_pipeline_autovc import data_gen_vc, data_gen_stft
import time, os
import utils
import h5py
import numpy as np
import mir_eval
import pandas as pd
from random import randint
import librosa
# import sig_process

import soundfile as sf

import matplotlib.pyplot as plt
from scipy.ndimage import filters


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


        stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = stat_file["feats_maximus"][()]
        min_feat = stat_file["feats_minimus"][()]


        stat_file.close()  

        with h5py.File(config.voice_dir+file_name, "r") as hdf5_file:
            mel = hdf5_file["world_feats"][()]

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



        out_mel = self.process_file(mel[:,:-2], speaker_index, speaker_index_2, sess)




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


        stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = stat_file["feats_maximus"][()]
        min_feat = stat_file["feats_minimus"][()]


        stat_file.close()  


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

        out_batches_mel = out_batches_mel*(max_feat[:-2] - min_feat[:-2]) + min_feat[:-2]

        return out_batches_mel



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """


        with tf.variable_scope('encoder') as scope:
            self.content_embedding_1 = modules.content_encoder(self.input_placeholder, self.speaker_onehot_labels, self.is_train)

        # if config.mode == "voc":
        with tf.variable_scope('decoder') as scope: 
            self.output_1 = modules.decoder(self.content_embedding_1, self.speaker_onehot_labels_1, self.is_train)
        with tf.variable_scope('post_net') as scope: 
            self.residual = modules.post_net(self.output_1, self.is_train)
            self.output = self.output_1 + self.residual
        with tf.variable_scope('encoder') as scope:
            scope.reuse_variables()
            self.content_embedding_2 = modules.content_encoder(self.output, self.speaker_onehot_labels, self.is_train)

class SSSynth_Content(Model):

    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        auto_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='post_net')

        self.auto_saver = tf.train.Saver(max_to_keep= config.max_models_to_keep, var_list = auto_var_list)

        # if not self.stft_var_list:
        self.encoder_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_encoder') 
        self.decoder_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_post_net') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'F0_Model') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Vuv_Model')

        self.encoder_saver = tf.train.Saver(max_to_keep= config.max_models_to_keep, var_list = self.encoder_var_list)

        self.decoder_saver = tf.train.Saver(max_to_keep= config.max_models_to_keep, var_list = self.decoder_var_list)

        sess.run(self.init_op)

        ckpt_auto = tf.train.get_checkpoint_state(config.log_dir)

        if ckpt_auto and ckpt_auto.model_checkpoint_path:
            print("Using the AUTOVC model in %s"%ckpt_auto.model_checkpoint_path)
            self.auto_saver.restore(sess, ckpt_auto.model_checkpoint_path)

        ckpt_encoder = tf.train.get_checkpoint_state(config.log_dir_encoder)

        if ckpt_encoder and ckpt_encoder.model_checkpoint_path:
            print("Using the STFT encoder model in %s"%ckpt_encoder.model_checkpoint_path)
            self.encoder_saver.restore(sess, ckpt_encoder.model_checkpoint_path)

        ckpt_decoder = tf.train.get_checkpoint_state(config.log_dir_decoder)

        if ckpt_decoder and ckpt_decoder.model_checkpoint_path:
            print("Using the STFT decoder model in %s"%ckpt_decoder.model_checkpoint_path)
            self.decoder_saver.restore(sess, ckpt_decoder.model_checkpoint_path)
    def save_model(self, sess, epoch, log_dir):
        """
        Save the model.
        """
        if config.use_speaker:
            checkpoint_file_encoder = os.path.join(config.log_dir_encoder, 'model.ckpt')
            self.encoder_saver.save(sess, checkpoint_file_encoder, global_step=epoch)

        checkpoint_file_decoder = os.path.join(config.log_dir_decoder, 'model.ckpt')
        self.decoder_saver.save(sess, checkpoint_file_decoder, global_step=epoch)
    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """

        self.optimizer = tf.train.AdamOptimizer(learning_rate = config.init_lr)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_f0 = tf.Variable(0, name='global_step_f0', trainable=False)
        self.global_step_vuv = tf.Variable(0, name='global_step_vuv', trainable=False)

        if config.use_speaker:
            self.harm_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_encoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_post_net')
        else:
            self.harm_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_decoder') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='stft_post_net')

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

        self.vuv_loss = tf.reduce_sum(tf.reduce_sum(binary_cross(self.vuv_placeholder, self.vuv)))

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

        self.get_summary(sess, config.log_dir_decoder)

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
                for feats_targs, stft_targs, targets_speakers, atbs in data_generator:
                    final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss, summary_str = self.train_model(feats_targs, stft_targs, targets_speakers, atbs, sess)
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
                    for feats_targs, stft_targs, targets_speakers, atbs in val_generator:


                        final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss, summary_str = self.validate_model(feats_targs, stft_targs, targets_speakers, atbs, sess)

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
                self.save_model(sess, epoch+1, config.log_dir_decoder)


    def train_model(self,feats_targs, stft_targs, targets_speakers, atbs, sess):
        """
        Function to train the model for each epoch
        """

        if config.f0_mode == "cont":
            feed_dict = {self.input_placeholder: feats_targs[:,:,:64], self.stft_placeholder: stft_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers, self.f0_placeholder: feats_targs[:,:,-2:-1], self.vuv_placeholder: feats_targs[:,:,-1:],self.is_train: True}
        elif config.f0_mode == "discrete":
            feed_dict = {self.input_placeholder: feats_targs[:,:,:64], self.stft_placeholder: stft_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers, self.f0_placeholder: atbs, self.vuv_placeholder: feats_targs[:,:,-1:],self.is_train: True}

        _,_,_, final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss = sess.run([self.final_train_function, self.f0_train_function, self.vuv_train_function, self.final_loss, self.recon_loss, self.recon_loss_0, self.content_loss, self.f0_loss, self.vuv_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, recon_loss, recon_loss_0, content_loss, f0_loss, vuv_loss, summary_str
 

    def validate_model(self,feats_targs, stft_targs, targets_speakers, atbs, sess):
        """
        Function to train the model for each epoch
        """

        if config.f0_mode == "cont":
            feed_dict = {self.input_placeholder: feats_targs[:,:,:64], self.stft_placeholder: stft_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers, self.f0_placeholder: feats_targs[:,:,-2:-1], self.vuv_placeholder: feats_targs[:,:,-1:],self.is_train: False}
        elif config.f0_mode == "discrete":
            feed_dict = {self.input_placeholder: feats_targs[:,:,:64], self.stft_placeholder: stft_targs, self.speaker_labels:targets_speakers, self.speaker_labels_1:targets_speakers, self.f0_placeholder: atbs, self.vuv_placeholder: feats_targs[:,:,-1:],self.is_train: False}
            
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


        stat_file = h5py.File('./stats_yam.hdf5', mode='r')

        max_feat = stat_file["feats_maximus"][()]
        min_feat = stat_file["feats_minimus"][()]


        stat_file.close()  

        with h5py.File(config.voice_dir+file_name, "r") as hdf5_file:
            mel = hdf5_file["world_feats"][()]

            voc_stft = hdf5_file["voc_stft"][()]

            back_stft = hdf5_file["back_stft"][()]

            stft = (voc_stft + back_stft)/2 

   

        return mel, stft


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
            out_featss = np.concatenate((out_mel[:feats.shape[0]], out_f0[:feats.shape[0]], feats[:,-1:]), axis = -1)

            audio_out = utils.feats_to_audio(out_featss) 

            audio_ori = utils.feats_to_audio(feats) 

            sf.write(os.path.join(config.output_dir,'./{}_output_3.wav'.format(file_name.split('/')[-1][:-4])), audio_out, config.fs)

            # except:
            #     unprocessable.append(file_name)

            count+=1

            utils.progress(count, len(file_list), "Files processed")

        print(unprocessable)


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

        if file_name.split('/')[-2][:-4].startswith('med'):
            mel, feats = self.read_med_file(file_name)
            acap_file = True

        else:
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
                f0_output = out_atb[:feats.shape[0], -1]
            else:
                f0_output = est_freq[:feats.shape[0]]

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

            audio_out = utils.feats_to_audio(np.concatenate((out_mel[:feats.shape[0]], feats[:,-2:]) , axis = -1))
            sf.write('./{}_ss_pred.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)

        elif config.f0_mode == "discrete":

            audio_out = utils.feats_to_audio(np.concatenate((out_mel[:feats.shape[0]], np.expand_dims(est_freq,-1)[:feats.shape[0]], feats[:,-1:]) , axis = -1))
            sf.write('./{}_ss_pred_dis.wav'.format(file_name.split('/')[-1][:-4]), audio_out, config.fs)

        if acap_file:

            audio = utils.feats_to_audio(feats) 
            sf.write('./{}_ori.wav'.format(file_name.split('/')[-1][:-4]), audio, config.fs)

        np.save(file_name.split('/')[-1][:-4], out_mel)





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


    def plot_features(self, stft, feats, out_feats, out_f0, out_vuv):

        plt.figure(1)


        ax1 = plt.subplot(311)

        plt.imshow(np.log(stft.T),aspect='auto',origin='lower')

        ax1.set_title("Input STFT", fontsize=10)

        ax1 = plt.subplot(312)

        plt.imshow(feats[:,:64].T,aspect='auto',origin='lower')

        ax1.set_title("Ground Truth Vocoder Features", fontsize=10)

        ax3 = plt.subplot(313, sharex = ax1, sharey = ax1)

        ax3.set_title("Output Vocoder Features", fontsize=10)

        plt.imshow(out_feats[:,:64].T,aspect='auto',origin='lower')


        plt.figure(4)

        ax1 = plt.subplot(211)

        plt.plot(feats[:,-1])

        ax1.set_title("Ground Truth VUV", fontsize=10)

        ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)

        plt.plot(out_vuv)

        ax1.set_title("Output VUV", fontsize=10)

        if config.f0_mode == "cont":

            plt.figure(2)
            f0_output = np.squeeze(out_f0)
            f0_output = f0_output*(1-feats[:,-1])
            f0_output[f0_output == 0] = np.nan
            # import pdb;pdb.set_trace()
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


    def process_file(self, mel, sess):


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
            speaker_index = config.singers.index('Bria')
            speaker_2 = np.repeat(speaker_index, config.batch_size)
            if config.use_speaker:
                feed_dict = {self.stft_placeholder: in_batch_mel, self.speaker_labels_1: speaker_2, self.is_train: False}
            else:
                feed_dict = {self.stft_placeholder: in_batch_mel, self.is_train: False}
            mel, f0, vuv = sess.run([self.output_stft, self.f0, self.vuv], feed_dict=feed_dict)
            out_batches_mel.append(mel)
            out_f0.append(f0)
            out_vuv.append(vuv)

        out_batches_mel = np.array(out_batches_mel)

        out_batches_mel = utils.overlapadd(out_batches_mel,nchunks_in)
        out_f0 = utils.overlapadd(np.array(out_f0), nchunks_in)
        out_vuv = utils.overlapadd(np.array(out_vuv), nchunks_in)

        if config.f0_mode == "cont":

            out_f0 = out_f0*(max_feat[-2] - min_feat[-2]) + min_feat[-2]



        out_batches_mel = out_batches_mel*(max_feat[:-2] - min_feat[:-2]) + min_feat[:-2]

        out_vuv = np.round(out_vuv)

        return out_batches_mel, out_f0, out_vuv



            #     import pdb;pdb.set_trace()



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
            self.content_embedding_stft = modules.content_encoder_stft(self.stft_placeholder, self.is_train)

        with tf.variable_scope('stft_decoder') as scope: 
            if config.use_speaker:
                self.output_stft_1 = modules.decoder_stft(self.content_embedding_stft, self.speaker_onehot_labels_1, self.is_train)
            else:
                self.output_stft_1 = modules.decoder_stft(self.content_embedding_stft, self.stft_placeholder, self.is_train)

        with tf.variable_scope('stft_post_net') as scope: 

            self.residual_stft = modules.post_net_stft(self.output_stft_1 , self.stft_placeholder, self.is_train)

            self.output_stft = self.output_stft_1 + self.residual_stft

        with tf.variable_scope('encoder') as scope:
            scope.reuse_variables()
            self.content_embedding_stft_2 = modules.content_encoder(self.output_stft,self.speaker_onehot_labels, self.is_train)

        with tf.variable_scope('F0_Model') as scope:
            # if config.enc_mode == "wave":
            # self.f0 = modules.nr_wavenet_f0(self.stft_placeholder, self.output_stft[:,:,:-4], self.output_stft[:,:,-4:], self.is_train)
            # elif config.enc_mode == "conv":
            self.f0 = modules.enc_dec_f0(self.stft_placeholder, self.is_train)
        with tf.variable_scope('Vuv_Model') as scope:
            # if config.enc_mode == "wave":
            # self.vuv = modules.nr_wavenet_vuv(self.stft_placeholder, self.output_stft[:,:,:-4], self.output_stft[:,:,-4:], self.f0, self.is_train)
            # elif config.enc_mode == "conv":
            self.vuv = modules.enc_dec_vuv(self.stft_placeholder, self.f0, self.output_stft[:,:,-4:], self.is_train)



def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = MultiSynth()
    model.train()

if __name__ == '__main__':
    test()





