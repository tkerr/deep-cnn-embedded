# mit_bih_afib_db.py
# Global objects and functions for use with the  
# MIT-BIH Atrial Fibrillation Database
# See https://physionet.org/content/afdb/1.0.0/

import os
import numpy as np
import pprint
from time import localtime, strftime

# ----------------------------------------------------------------------
# Globals.
# ----------------------------------------------------------------------

LOCAL_DATA_PATH = r'E:\Data\MIT-BIH-AFIB\files'
LOCAL_TFRECORD_PATH = r'E:\Data\MIT-BIH-AFIB\tfrecord'

# Physionet top-level directory used for streaming data records
PN_DIR_BASE = 'afdb/1.0.0/'

PATIENT_IDS = [ 
    '04015', '04043', '04048', '04126', '04746', '04908', '04936', '05091',
    '05121', '05261', '06426', '06453', '06995', '07162', '07859', '07879',
    '07910', '08215', '08219', '08378', '08405', '08434', '08455']
NUM_PATIENTS = len(PATIENT_IDS)

FS_HZ = 250   # ECG sample rate in Hz

LABELS = {'Q':0, 'N':1, 'AFL':2, 'AFIB':3, 'J':4}  # Data labels, SAME ORDER as Icentia11k
CLASS_NAMES = list(LABELS.keys())                  # Class names in same order as labels
NUM_CLASSES = len(LABELS)
RTYPES = ['AFIB', 'AFL', 'J', 'N', 'Q']            # Rhythm types in alphabetical order

TFRECORD_FILE_RE = r'(\d{5})_([A-Z]+)_(\d{7})_(\d{7})\.tfrecord' # Regular expression string

pp = pprint.PrettyPrinter(indent=2, width=120) # For printing lists, dictionaries, etc.

# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def get_rhythms_file(rtype):
    """
    Create and return a rhythms CSV file name from the rhythm type.
    """
    file_name = os.path.join(LOCAL_TFRECORD_PATH, 'rhythms_{}.csv'.format(rtype))
    return file_name

# ----------------------------------------------------------------------
def get_sequences_file(rtype):
    """
    Create and return a sequences CSV file name from the rhythm type.
    """
    file_name = os.path.join(LOCAL_TFRECORD_PATH, 'sequences_{}.csv'.format(rtype))
    return file_name

# ----------------------------------------------------------------------
def get_ordered_file(rtype):
    """
    Create and return an ordered sequences CSV file name from the rhythm type.
    """
    file_name = os.path.join(LOCAL_TFRECORD_PATH, 'ordered_{}.csv'.format(rtype))
    return file_name

# ----------------------------------------------------------------------
def get_csv_record(pid, rtype, start, length):
    """
    Create and return a formatted CSV record.
    """
    csv_record = '{},{},{},{}\n'.format(pid, rtype, start, length)
    return csv_record

# ----------------------------------------------------------------------
def get_wfdb_path(pid):
    """
    Create and return a WFDB path name from a patient ID.
    """
    path = os.path.join(LOCAL_DATA_PATH, '{:05d}'.format(int(pid)))
    return path

# ----------------------------------------------------------------------
def get_tfrecord_filename(pid, rtype, start, length):
    """
    Create and return a TFRecord path and file name from component parts.
    """
    pid_str = '{:05d}'.format(int(pid))
    file_path = os.path.join(LOCAL_TFRECORD_PATH, pid_str)
    file_name = '{}_{}_{:07d}_{:07d}.tfrecord'.format(pid_str, rtype.upper(), int(start), int(length))
    file_spec = os.path.join(file_path, file_name)
    return file_path, file_spec

# ----------------------------------------------------------------------
def num_samples(time_sec):
    """
    Return the number of samples for the given ECG length in seconds.
    """
    global FS_HZ
    return int(FS_HZ * time_sec)
    
# ----------------------------------------------------------------------
def timestamp():
    """
    Convenience function to return a formatted timestamp.
    """
    return strftime('%Y-%m-%d %H:%M:%S', localtime())

# ----------------------------------------------------------------------
def pprint(object):
    """
    Convenience function to pretty print an object (list, dictionary, etc.)
    """
    global pp
    pp.pprint(object)

