import numpy as np
import tensorflow as tf
import os


wav_dir_yam = '../datasets/Data/'


modes = ['wave', 'vocoder', 'vocoderf0', 'logmel']
encoder_modes = ['lstm', 'wave', 'conv']
op_mode = "sep_content"
output_dir = '../sep_content/output_med_last/'

mode = 'vocoder'
encoder_mode = 'lstm'
f0_mode = "cont"
use_speaker = True

fs = 32000
nfft = 1024
hopsize = 160
hoptime = hopsize/fs
window = np.hanning(nfft)



filter_len = 5
encoder_layers = 4
filters = 32

# mixture_encoding = 32

sep_filters = 128

augment_filters_every = 2

f_max = 7600
f_min = 125
min_level_db= -100
num_mels = 80
ref_level_db = 20

voice_dir = '../consolidated_research/voice_32/'
backing_dir = './backing/'



#CQT Parameters
fmin = 32.70
bins_per_octave = 60
# bins_per_octave = 24
n_octaves = 6
cqt_bins = bins_per_octave*n_octaves

energy_threshold = 0.01
ms_sil = 2000

world_offset = -20.0


lstm_size = 32
out_lstm = 512
code_sam = 16
filter_len = 5

med_to_use = ['MusicDelta_Gospel.wav', 'FacesOnFilm_WaitingForGa.wav', 'Lushlife_ToynbeeSuite.wav', 'HezekiahJones_BorrowedHeart.wav', 'BrandonWebster_DontHearAThing.wav', 'AimeeNorwich_Child.wav', 'ClaraBerryAndWooldog_AirTraffic.wav']


output_features = 64
stft_features = 513


log_dir = "log_avc_{}_{}".format(mode, encoder_mode)

if use_speaker:
	if f0_mode == "cont":
		log_dir_encoder = "log_avc_{}_{}_encoder/".format(mode, encoder_mode)
		log_dir_decoder = "log_avc_{}_{}_decoder_speaker/".format(mode, encoder_mode)
	elif f0_mode == "discrete":
		log_dir_encoder = "log_avc_{}_{}_encoder_discrete/".format(mode, encoder_mode)
		log_dir_decoder = "log_avc_{}_{}_decoder_speaker_discrete/".format(mode, encoder_mode)

else:
	if f0_mode == "cont":
		log_dir_encoder = "log_avc_{}_{}_encoder/".format(mode, encoder_mode)
		log_dir_decoder = "log_avc_{}_{}_decoder_stft/".format(mode, encoder_mode)
	elif f0_mode == "discrete":
		log_dir_encoder = "log_avc_{}_{}_encoder_discrete/".format(mode, encoder_mode)
		log_dir_decoder = "log_avc_{}_{}_decoder_stft_discrete/".format(mode, encoder_mode)





first_conv = 10
wavenet_layers = 5
wavenet_filters = 256



split = 0.9
singers = ['Kasai', 'HiyamaKiyoshi', 'Yasumura', 'Yamamoto', 'Shirai', 'Kikuchi', 'Hayashi', 'Inoue', 'Matsushima', 'Honma', 'Takeyama', 'Someya', 'Hasegawa', 'Terada', 'KokubuYurie', 'Kaneko', 'Yoshida', 'Gonda', 'Izawa', 'Koizumi', 'MatsudaRere', 'Mishina', 'Fuga', 'Kon', 'YamaokaKyoko', 'Ebony', 'Shimogawa', 'Nabeshima', 'Ozaki', 'Mochizuki', 'Nikolas', 'Katsura', 'Sanada', 'Miyazaki', 'Nishizawa', 'Scott', 'Ogawa', 'Ishikawa', 'Matsumoto', 'Takahashi', 'Itoh', 'Kojima', 'Bria', 'Kaidoh', 'Iino', 'Gary']
unprocessable = ['yam_Gary_008_1.hdf5', 'yam_Gary_002_0.hdf5', 'yam_HiyamaKiyoshi_015_1.hdf5', 'yam_HiyamaKiyoshi_015_2.hdf5', 'yam_HiyamaKiyoshi_004_0.hdf5']


num_speakers = len(singers)


mu = 1 
lamda = 1

num_epochs = 2500

batches_per_epoch_train = 100
batches_per_epoch_val = 10

f0_threshold = 0.5

batch_size = 30
samples_per_file = 6
if mode =="wave":
	max_phr_len = 128*hopsize
else:
	max_phr_len = 128



init_lr = 0.001

max_models_to_keep = 10

print_every = 1
save_every = 50
validate_every = 1
