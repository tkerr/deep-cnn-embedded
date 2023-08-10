# icentia11k_tfrecord.py
# Functions to create TensorFlow TFRecord datasets from the Icentia11k data.

# System packages.
import numpy as np
import os
import re
import tensorflow as tf

# Local packages.
import fileutils as fu
import icentia11k as ic


# ----------------------------------------------------------------------
# Global objects.
# ----------------------------------------------------------------------

ECG_LENGTH = 7500
ECG_OFFSET = 1250
USE_ZNORM = False

train_list = []
test_list = []
val_list = []


# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------
def create_tfrecord_lists(
    tfrecord_path, 
    master_csv_path,
    test_split=0.1, 
    val_split=0.1, 
    max_records=0,
    seed=None):
    """
    Create file lists for train, test and validation data.
    Selects records from a set of master ordered CSV files.
    
    Parameters:
        tfrecord_path : str
            Top-level directory path to tfrecord files
        master_csv_path : str
            Path to master ordered CSV files used to select examples
        test_split : float
            Fraction of data to use for test (0.0 - 1.0)
        val_split : float
            Fraction of data to use for validation (0.0 - 1.0)
        max_records : int
            The maximum number of total records to include in the lists,
            i.e. len(train_list) + len(test_list) + len(val_list)
            Ignored if zero (default)
        seed : int
            Random number generator seed value
    Returns:
        train_list, test_list, val_list
            Lists of tfrecord filenames for training, testing and validation
    """
    global ECG_LENGTH
    global ECG_OFFSET
    global train_list
    global test_list
    global val_list
    
    rng = np.random.default_rng(seed=seed)
    tfrecord_count = 0
    
    train_list = []
    test_list = []
    val_list = []
    
    # Open the master ordered CSV files.
    master_fd_list = [None for r in ic.RTYPES]
    i = 0
    for rtype in ic.RTYPES:
        file = ic.get_master_filename(rtype)
        filespec = os.path.join(master_csv_path, file)
        master_fd_list[i] = fu.open_file(filespec)
        i += 1
    
    # Select records uniformly from each of the CSV files.
    # Place into the training list.
    select_done = False
    tfrecord_count = 0
    while not select_done:
        for fd in master_fd_list:
            line = fd.readline().strip()
            if (len(line) > 0):
                (pid, sid, tfrtype, start, _) = line.split(',') # Ignoring length in record...
                tfstart = int(start) + ECG_OFFSET
                filename = ic.get_tfrecord_filename(pid[1:], sid[1:], tfrtype, tfstart, ECG_LENGTH) # ...Using ECG_LENGTH instead
                topdir, pid_dir = ic.get_pid_dirs(pid[1:])
                tfrecord_file = os.path.join(tfrecord_path, topdir, pid_dir, filename)
                train_list.append(tfrecord_file)
                tfrecord_count += 1
            else:
                # Out of records.
                select_done = True
        if (max_records > 0):
            if (tfrecord_count >= max_records):
                select_done = True
    
    # Close the master ordered CSV files.
    for fd in master_fd_list:
        fu.close_file(fd)
    
    # Create an array of random indices into the train array.
    rand_idx = rng.choice(tfrecord_count, size=tfrecord_count, replace=False)
    
    # Initialize test and validation lists.
    test_count = int(round(tfrecord_count * test_split / 4.0, 0))
    val_count = int(round(tfrecord_count * val_split / 4.0, 0))

    test_afib_count = 0
    test_afl_count = 0
    test_n_count = 0
    test_q_count = 0

    val_afib_count = 0
    val_afl_count = 0
    val_n_count = 0
    val_q_count = 0
    
    # Create the test and validation lists.
    for idx in rand_idx:
        file = train_list[idx]
        if '_AFIB_' in file:
            if (test_afib_count < test_count):
                test_list.append(file)
                test_afib_count += 1
            elif (val_afib_count < val_count):
                val_list.append(file)
                val_afib_count += 1
        elif '_AFL_' in file:
            if (test_afl_count < test_count):
                test_list.append(file)
                test_afl_count += 1
            elif (val_afl_count < val_count):
                val_list.append(file)
                val_afl_count += 1
        elif '_N_' in file:
            if (test_n_count < test_count):
                test_list.append(file)
                test_n_count += 1
            elif (val_n_count < val_count):
                val_list.append(file)
                val_n_count += 1
        elif '_Q_' in file:
            if (test_q_count < test_count):
                test_list.append(file)
                test_q_count += 1
            elif (val_q_count < val_count):
                val_list.append(file)
                val_q_count += 1

    # Remove test files from the train list.        
    for file in test_list:
        train_list.remove(file)
        
    # Remove validation files from the train list.
    for file in val_list:
        train_list.remove(file)

    return train_list, test_list, val_list

# ----------------------------------------------------------------------
def get_tfrecord_lists(
    train_file,
    test_file,
    val_file,
    path=''):
    """
    Create file lists for train, test and validation data from text files.
    The text files contain a list of TFRecord files for each set.
    
    Parameters:
        train_file : str
            The training file name used to load train_list
        test_file : str
            The test file name used to load test_list
        val_file : str
            The validation file name used to load val_list
        path : str
            Optional path prefix
    Returns:
        train_list, test_list, val_list
            Lists of tfrecord filenames for training, testing and validation
    """
    global train_list
    global test_list
    global val_list
    
    train_list = []
    test_list = []
    val_list = []
    
    fd = fu.open_file(train_file)
    for line in fd:
        train_list.append(os.path.join(path, line.strip()))
    fu.close_file(fd)
    
    fd = fu.open_file(test_file)
    for line in fd:
        test_list.append(os.path.join(path, line.strip()))
    fu.close_file(fd)
    
    fd = fu.open_file(val_file)
    for line in fd:
        val_list.append(os.path.join(path, line.strip()))
    fu.close_file(fd)
    
    return train_list, test_list, val_list
    
# ----------------------------------------------------------------------
def ecg_parse_fn(serialized):
    """
    Mapping function used to parse a tfrecord.
    
    Global variables ECG_LENGTH and USE_ZNORM must be set.
    
    Returns a tuple of (ecg, one_hot), where one-hot is a 4-element vector.
    """
    global ECG_LENGTH
    global USE_ZNORM
    features = tf.io.parse_single_example(
        serialized,
        features={
            'ecg_raw': tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_znorm': tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            #'label': tf.io.FixedLenFeature([], tf.int64)
            'one_hot': tf.io.FixedLenFeature([4], tf.float32),
    })
    if USE_ZNORM:
        ecg = features['ecg_znorm']
    else:
        ecg = features['ecg_raw']
    #label = tf.cast(features['label'], tf.int32)
    one_hot = features['one_hot']
    return (ecg, one_hot)


# ----------------------------------------------------------------------
# Classes.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Main program used as an example for test and debug.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print('Ictntia11k_tfrecord test program not implemented.')
    
