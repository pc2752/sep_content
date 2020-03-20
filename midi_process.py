import os,re
import numpy as np
import vamp
import re
import matplotlib.pyplot as plt


from scipy.stats import norm

import config

def coarse_code(x, n_states = 3, sigma = 0.4):
    """Coarse-code value to finite number of states, each with a Gaussian response.

    Parameters
    ----------
    x : ndarray
        Vector of normalized values [0.0;1.0], shape (nframes,).
    n_states : int
        Number of states to use for coase coding.
    sigma : float
        Sigma (scale, standard deviation) parameter of normal distribution 
        used internally to perform coarse coding. Default: 0.4

    Returns
    -------
    ndarray
        Matrix of shape (nframes, n_states).

    See also
    --------
    https://en.wikipedia.org/wiki/Neural_coding#Position_coding
    https://plus.google.com/+IlyaEdrenkin/posts/B55jf3wUBvD
    https://github.com/CSTR-Edinburgh/merlin/blob/master/src/frontend/label_normalisation.py
    """
    assert np.all(x >= 0.0) and np.all(x <= 1.0), 'expected input to be normalized in range [0;1]'
    mu = np.linspace(0.0, 1.0, num=n_states, endpoint=True)
    return np.hstack([norm.pdf(x, mu_k, sigma).reshape((-1, 1)) for mu_k in mu]) 


def note_str_to_num(note, base_octave=-1):
    """Convert note pitch as string to MIDI note number."""
    patt = re.match('^([CDEFGABcdefgab])([b#]*)(-?)(\d+)$', note)
    if patt is None:
        raise ValueError('invalid note string "{}"'.format(note))
    base_map = {'C': 0,
                'D': 2,
                'E': 4,
                'F': 5,
                'G': 7,
                'A': 9,
                'B': 11}
    base, modifiers, sign, octave = patt.groups()
    base_num = base_map[base.upper()]
    mod_num = -modifiers.count('b') + modifiers.count('#')
    sign_mul = -1 if sign == '-' else 1
    octave_num = 12*int(octave)*sign_mul - 12*base_octave
    note_num = base_num + mod_num + octave_num
    if note_num < 0 or note_num >= 128:
        raise ValueError('note string "{}" resulted in out-of-bounds note number {:d}'.format(note, note_num))
    return note_num


def note_num_to_str(note, base_octave=-1):
    """Convert MIDI note number to note pitch as string."""
    base = note % 12
    # XXX: base_map should probably depend on key
    base_map = ['C',
                'C#',
                'D',
                'D#',
                'E',
                'F',
                'F#',
                'G',
                'G#',
                'A',
                'A#',
                'B']
    base_note = note%12
    octave = int(np.floor(note/12)) + base_octave
    return '{}{:d}'.format(base_map[base_note], octave)

def rock(audio):
    jojo = vamp.collect(audio, config.fs, "pyin:pyin", step_size=config.hopsize, output="notes")

    import pdb;pdb.set_trace()
def open_f0_file(filename):
    """
    Returns a numpy array with the start-time, end-time and notes from the f0 file
    """
    with open(filename, "r") as lab_f:
        phos = lab_f.readlines()
        phos2 = [x.split() for x in phos]
        popo = [float(x[0]) for x in phos2]
        diff = popo[1] - popo[0]
        phos3 = np.array([[float(x[0]), float(x[0]) + diff, float(x[1])] for x in phos2])
    return phos3

def process_lab_file(filename, stft_len, div_factor, pho_list):

    lab_f = open(filename)

    # note_f=open(in_dir+lf[:-4]+'.notes')
    phos = lab_f.readlines()
    lab_f.close()

    phonemes=[]

    for pho in phos:
        st,end,phonote=pho.split()
        st = int(np.round(float(st)/div_factor))
        en = int(np.round(float(end)/div_factor))
        if phonote=='pau' or phonote=='br' or phonote == 'sil':
            phonote='Sil'
        phonemes.append([st,en,phonote])


    strings_p = np.zeros((phonemes[-1][1],6))

    prev = pho_list.index('Sil')

    for i in range(len(phonemes)):
        pho=phonemes[i]
        if not i == len(phonemes)-1:
            npho = phonemes[i+1]
            next_pho = pho_list.index(npho[2])
        else:
            next_pho = pho_list.index('Sil')
        value = pho_list.index(pho[2])
        context = coarse_code(np.linspace(0.0,1.0, len(strings_p[pho[0]:pho[1]+1,0])))
        strings_p[pho[0]:pho[1]+1,0] = prev
        prev = value
        strings_p[pho[0]:pho[1]+1,1] = value
        strings_p[pho[0]:pho[1]+1,2] = next_pho
        strings_p[pho[0]:pho[1]+1,3:] = context

    return strings_p


def process_notes_file(filename, stft_len, div_factor):

    lab_f = open(filename)
    # note_f=open(in_dir+lf[:-4]+'.notes')
    phos = lab_f.readlines()
    lab_f.close()

    phonemes=[]

    for pho in phos:
        st,end,phonote=pho.split()
        note, combo = phonote.split('/p:')
        if note == 'xx':
            note_num = 0
        else:
            note_num = note_str_to_num(note)

        st = int(np.round(float(st)/config.hoptime)/div_factor)
        en = int(np.round(float(end)/config.hoptime)/div_factor)
        # if phonote=='pau' or phonote=='br':
        #     phonote='sil'
        phonemes.append([st,en,note_num, combo])

    strings_p = np.zeros((phonemes[-1][1],6))
    strings_c = np.zeros((phonemes[-1][1],6))

    prev = 0

    for i in range(len(phonemes)):
        if not i == len(phonemes)-1:
            npho = phonemes[i+1]
            next_pho = config.notes.index(npho[2])
        else:
            next_pho = 0
        pho=phonemes[i]
        value = config.notes.index(pho[2])

        
        context = coarse_code(np.linspace(0.0,1.0, len(strings_p[pho[0]:pho[1]+1,0])))
        strings_p[pho[0]:pho[1]+1, 1] = value
        strings_p[pho[0]:pho[1]+1, 0] = prev
        strings_p[pho[0]:pho[1]+1, 2] = next_pho
        for j, p in enumerate(pho[3].split('-')):
            strings_c[pho[0]:pho[1] + 1, j+1] = config.phonemas.index(p)+1
        strings_p[pho[0]:pho[1]+1,3:] = context
        prev = value
    return strings_p, strings_c.reshape(-1,6)
