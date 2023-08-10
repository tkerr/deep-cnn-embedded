# wfdb_utils.py
# Waveform database (wfdb) utility functions

# System packages.
import os
import wfdb
import numpy as np

import fileutils as fu


# ----------------------------------------------------------------------
# Global objects.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Function to parse a WFDB file path.
def _parse_file_path(file_path):
    path = os.path.dirname(os.path.abspath(file_path))
    basename = os.path.basename(file_path)
    return path, basename

# ----------------------------------------------------------------------
# Function to read annotation data in WFDB format.
def read_annotation(file_path, start=0, length=None, pn_dir=None):
    ann = None
    end = None
    if length is not None:
        end = start + length
    try:
        ann = wfdb.rdann(file_path, 'atr', sampfrom=start, sampto=end, shift_samps=True, pn_dir=pn_dir)
    except Exception as err:
        print('Annotation read error: {}'.format(str(err)))
        print('File: {}'.format(file_path))
    return ann

# ----------------------------------------------------------------------
# Function to read header data in WFDB format.
def read_header(file_path, pn_dir=None):
    hdr = None
    try:
        hdr = wfdb.rdheader(file_path, pn_dir=pn_dir)
    except Exception as err:
        print('Header read error: {}'.format(str(err)))
        print('File: {}'.format(file_path))
    return hdr
    
# ----------------------------------------------------------------------
# Function to read sample data in WFDB format.
def read_data(file_path, start=0, length=None, pn_dir=None):
    rec = None
    end = None
    if length is not None:
        end = start + length
    try:
        rec = wfdb.rdrecord(file_path, sampfrom=start, sampto=end, pn_dir=pn_dir)
    except Exception as err:
        print('Data read error: {}'.format(str(err)))
        print('File: {}'.format(file_path))
    return rec

# ----------------------------------------------------------------------
# Function to write annotation locally in WFDB format.
def write_annotation(ann, file_path):
    ok = False
    path, basename = _parse_file_path(file_path)
    fu.mkpath(path)  # Create paths if needed
    ann.record_name = basename
    
    # Must ensure at least one label exists or the write function will fail.
    if (ann.ann_len == 0):
        # Add a bogus 'Learning' label. See wfdb.show_ann_labels().
        ann.ann_len = 1
        ann.aux_note = ['None']
        ann.chan = np.zeros(1, dtype=np.int32)
        ann.num = np.zeros(1)
        ann.sample = np.zeros(1, dtype=np.int64)
        ann.subtype = np.zeros(1)
        ann.symbol = ['?']
    
    try:
        ann.wrann(write_fs=True, write_dir=path)
        ok = True
    except Exception as err:
        print('Annotation write error: {}'.format(str(err)))
        print('File: {}'.format(basename))
    return ok

# ----------------------------------------------------------------------
# Function to write sample data and header locally in WFDB format.
def write_data(rec, file_path):
    ok = False
    path, basename = _parse_file_path(file_path)
    fu.mkpath(path)  # Create paths if needed
    
    # Must ensure ADC signal exists or the write function will fail.
    if rec.d_signal is None:
        rec.d_signal = rec.adc(expanded=False, inplace=False)
    rec.file_name = ['{}.dat'.format(basename) for i in range(rec.n_sig)]
    rec.record_name = basename
    
    try:
        rec.wrsamp(expanded=False, write_dir=path)
        ok = True
    except Exception as err:
        print('Data write error: {}'.format(str(err)))
        print('File: {}'.format(basename))
    return ok
    
# ----------------------------------------------------------------------
# Function to parse the waveform header and annotation and return a list 
# of tuples indicating onset and offset of labeled rhythms.
# Returns a list of waveform count tuples: (symbol, start, length)
def parse_waveforms(ann, hdr):
    wav_list = []
    wav_start = -1
    wav_end = -1
    wav_type = '?'
    in_waveform = False
    
    for i in range(len(ann.aux_note)):
        if ann.aux_note[i].startswith('('):
            if in_waveform:
                # Found another '(' before finding a ')'
                wav_end = ann.sample[i]
                wav_len = wav_end - wav_start
                wav_list.append((wav_type, wav_start, wav_len))
            wav_start = ann.sample[i]
            wav_type = ann.aux_note[i][1:]
            in_waveform = True
        elif ann.aux_note[i].startswith(')'):
            wav_end = ann.sample[i]
            wav_len = wav_end - wav_start
            wav_list.append((wav_type, wav_start, wav_len))
            in_waveform = False
        
    if in_waveform:
        wav_len = hdr.sig_len - wav_start
        wav_list.append((wav_type, wav_start, wav_len))
    return wav_list
