# icentia11k.py
# Global objects and functions for use with the  
# Icentia11k Single Lead Continuous Raw Electrocardiogram Dataset
# See: https://physionet.org/content/icentia11k-continuous-ecg/1.0/

import numpy as np
import pprint
from time import localtime, strftime

# ----------------------------------------------------------------------
# Globals.
# ----------------------------------------------------------------------

LOCAL_DATA_PATH = r'E:/Data/Icentia11k/data'
LOCAL_TFRECORD_PATH = r'E:/Data/Icentia11k/tfrecord'

# Physionet top-level directory used for streaming data records
PN_DIR_BASE = 'icentia11k-continuous-ecg/1.0/'

NUM_PATIENTS = 11000     # Number of patients; Patient ID (PID) range is 0 - 10999
MAX_SEGMENTS = 50        # Max segments per PID; Segment ID (SID) range is 0 - 49
MAX_ECG_LENGTH = 1048577 # Max ECG length in samples (2^20 + 1)
FS_HZ = 250              # ECG sample rate in Hz

# CSV sequence file base names, regular expression and format strings
ORDERED_MASTER_BN = 'ordered_master'  # File base name per rhythm type
ORDERED_SUBDIR_BN = 'ordered_dir'     # File base name per subdir and rhythm type
RHYTHMS_FILE_RE = r'rhythms_(p\d{5}_p\d{5})\.csv'              # Regular expression string
SEQUENCES_FILE_RE = r'sequences_(p\d{5}_p\d{5})_([A-Z])+\.csv' # Regular expression string
SEQUENCES_FILE_FMT = r'sequences_p{:05d}_p{:05d}_{}.csv'       # Format string

# WFDB data and TFRecord file names
WFDB_DAT_FILE_RE = r'(p\d{5})_(s\d{2})_([A-Z]+)_(\d{7})_(\d{7})\.dat'      # Regular expression string
WFDB_DAT_FILE_FMT = r'p{:05d}_s{:02d}_{}_{:07d}_{:07d}.dat'                # Format string
WFDB_BN_FILE_FMT = r'p{:05d}_s{:02d}_{}_{:07d}_{:07d}'                     # Format string, no file extension
TFRECORD_FILE_RE = r'(p\d{5})_(s\d{2})_([A-Z]+)_(\d{7})_(\d{7})\.tfrecord' # Regular expression string
TFRECORD_FILE_FMT = r'p{:05d}_s{:02d}_{}_{:07d}_{:07d}.tfrecord'           # Format string

LABELS = {'Q':0, 'N':1, 'AFL':2, 'AFIB':3}  # Data labels
CLASS_NAMES = list(LABELS.keys())           # Class names in same order as labels
NUM_CLASSES = len(LABELS)
RTYPES = ['AFIB', 'AFL', 'N', 'Q']          # Rhythm types in alphabetical order
SUBDIRS = ['p00', 'p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10']

pp = pprint.PrettyPrinter(indent=2, width=120) # For printing lists, dictionaries, etc.

# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------

def num_samples(time_sec):
    """
    Return the number of samples for the given ECG length in seconds.
    """
    global FS_HZ
    return int(FS_HZ * time_sec)

# ----------------------------------------------------------------------
def get_master_filename(rtype):
    """
    Return the master ordered CSV file name for the specified rhythm type.
    """
    global ORDERED_MASTER_BN
    filename = '{}_{}.csv'.format(ORDERED_MASTER_BN, rtype.upper())
    return filename
    
# ----------------------------------------------------------------------
def get_pid_dirs(pid):
    """
    Return the top-level and patient ID directories for the specified Patient ID.
    Example: pid = 123, topdir = 'p00', pid_dir = 'p00123'
    """
    pid_dir = 'p{:05d}'.format(int(pid))
    topdir = pid_dir[:3]
    return topdir, pid_dir

# ----------------------------------------------------------------------
def get_tfrecord_filename(pid, sid, rtype, start, length):
    """
    Return the TFRecord file name for the specified input.
    """
    global TFRECORD_FILE_FMT
    basename = TFRECORD_FILE_FMT.format(
        int(pid), int(sid), rtype.upper(), int(start), int(length))
    return basename

# ----------------------------------------------------------------------
def get_wfdb_basename(pid, sid, rtype, start, length):
    """
    Return the waveform database (WFDB) file base name for the specified input.
    """
    global WFDB_BN_FILE_FMT
    if (str(pid)[0] == 'p'):
        i_pid = int(pid[1:])
    else:
        i_pid = int(pid)
    if (str(sid)[0] == 's'):
        i_sid = int(sid[1:])
    else:
        i_sid = int(sid)
    basename = WFDB_BN_FILE_FMT.format(
        i_pid, i_sid, rtype.upper(), int(start), int(length))
    return basename

# ----------------------------------------------------------------------
def get_histo_stats(pid_histo):
    """
    Compute statistics on a histogram of Patient IDs.
    PIDs with a zero count are not included in the statistics.
    """
    histo_sum = np.sum(pid_histo)
    histo_cnt = np.count_nonzero(pid_histo)
    histo_nz  = pid_histo[pid_histo != 0]
    histo_min = np.min(histo_nz)
    histo_avg = np.average(histo_nz)
    histo_max = np.max(histo_nz)
    return histo_sum, histo_cnt, histo_min, histo_avg, histo_max
    
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

