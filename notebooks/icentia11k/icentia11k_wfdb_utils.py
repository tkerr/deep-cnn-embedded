# icentia11k_wfdb_utils.py
# Waveform database (wfdb) utility functions
# Designed for use with the Icentia11k dataset
# See https://physionet.org/content/icentia11k-continuous-ecg/1.0/


# System packages.
import os
import sys
import wfdb
import numpy as np

import fileutils as fu
import icentia11k as ic


# ----------------------------------------------------------------------
# Global objects.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Function to get a dataset filename and pn_dir from patient ID and segment ID.
def get_dataset_filename(patient_id, segment_id, stream):
    topdir, pid_dir = ic.get_pid_dirs(patient_id)
    basename = '{}_s{:02d}'.format(pid_dir, int(segment_id))
    if stream:
        filename = basename
        pn_dir = '{}/{}/{}'.format(ic.PN_DIR_BASE, topdir, pid_dir)
    else:
        filename = '{}/{}/{}/{}'.format(ic.LOCAL_DATA_PATH, topdir, pid_dir, basename)
        pn_dir = ic.PN_DIR_BASE # Not used for local files
    #print(filename)  
    return filename, pn_dir

# ----------------------------------------------------------------------
# Function to get local path and filename from patient ID, segment ID, rhythm type start and length.
def get_local_filename(patient_id, segment_id, rtype, start, length):
    topdir, pid_dir = ic.get_pid_dirs(patient_id)
    path = os.path.join(ic.LOCAL_DATA_PATH, topdir, pid_dir)
    basename = ic.get_wfdb_basename(
        patient_id, segment_id, rtype, start, length)
    return (path, basename)

# ----------------------------------------------------------------------
# Function to read annotation data in WFDB format.
def read_annotation(patient_id, segment_id, *, start=0, length=ic.MAX_ECG_LENGTH, stream=False):
    ann = None
    end = start + length
    if (end > ic.MAX_ECG_LENGTH):
        end = ic.MAX_ECG_LENGTH
    filename, pn_dir = get_dataset_filename(patient_id, segment_id, stream)
    try:
        if stream:
            ann = wfdb.rdann(filename, 'atr', sampfrom=start, sampto=end, shift_samps=True, pn_dir=pn_dir)
        else:
            ann = wfdb.rdann(filename, 'atr', sampfrom=start, sampto=end, shift_samps=True)
    except Exception as err:
        print('Annotation read error: {}'.format(str(err)))
        print('File: {}'.format(filename))
    return ann

# ----------------------------------------------------------------------
# Function to read ECG data in WFDB format.
def read_ecg(patient_id, segment_id, *, start=0, length=ic.MAX_ECG_LENGTH, stream=False):
    rec = None
    end = start + length
    if (end > ic.MAX_ECG_LENGTH):
        end = ic.MAX_ECG_LENGTH
    filename, pn_dir = get_dataset_filename(patient_id, segment_id, stream)
    try:
        if stream:
            rec = wfdb.rdrecord(filename, sampfrom=start, sampto=end, pn_dir=pn_dir)
        else:
            rec = wfdb.rdrecord(filename, sampfrom=start, sampto=end)
    except Exception as err:
        print('ECG read error: {}'.format(str(err)))
        print('File: {}'.format(filename))
    return rec

# ----------------------------------------------------------------------
# Function to write annotation locally in WFDB format.
def write_annotation(ann, patient_id, segment_id, rtype, start, length):
    ok = False
    (path, basename) = get_local_filename(patient_id, segment_id, rtype, start, length)
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
# Function to write ECG data and header locally in WFDB format.
def write_ecg(ecg, patient_id, segment_id, rtype, start, length=None):
    ok = False
    if length is None:
        length = ecg.sig_len
    (path, basename) = get_local_filename(patient_id, segment_id, rtype, start, length)
    fu.mkpath(path)  # Create paths if needed
    
    # Must perform an ADC operation or the write function will fail.
    ecg.d_signal = ecg.adc(expanded=False, inplace=False)
    ecg.file_name = ['{}.dat'.format(basename)]
    ecg.record_name = basename
    try:
        ecg.wrsamp(expanded=False, write_dir=path)
        ok = True
    except Exception as err:
        print('ECG write error: {}'.format(str(err)))
        print('File: {}'.format(basename))
    return ok

