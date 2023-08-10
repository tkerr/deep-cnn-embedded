# mit_bih_afib_tfrecord.py
# Functions to create TensorFlow TFRecord datasets from the 
# MIT-BIH Atrial Fibrillation Database
# See https://physionet.org/content/afdb/1.0.0/

# System packages.
import numpy as np
import os
import re
import tensorflow as tf

# Local packages.
import fileutils as fu
import mit_bih_afib_db as db


# ----------------------------------------------------------------------
# Global objects.
# ----------------------------------------------------------------------

ECG_LENGTH = 7500           # ECG length in samples
ECG_FEATURE = 'ecg_fir_z'   # Selected TFRecord feature

train_list = []
test_list = []
val_list = []


# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------
def create_tfrecord_lists(
    rtype_list,
    test_split=0.1, 
    val_split=0.1, 
    max_records=0,
    seed=None):
    """
    Create file lists for train, test and validation data.
    Selects records from a set of ordered CSV files.
    
    Parameters:
        rtype_list : list
            List of rhythm type strings to include in the datasets.
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
    global train_list
    global test_list
    global val_list
    
    rng = np.random.default_rng(seed=seed)
    tfrecord_count = 0
    num_classes = len(rtype_list)
    
    train_list = []
    test_list = []
    val_list = []
    
    # Open the ordered CSV files.
    ordered_fd_list = [None for r in rtype_list]
    i = 0
    for rtype in rtype_list:
        file = db.get_ordered_file(rtype)
        ordered_fd_list[i] = fu.open_file(file)
        i += 1
    
    # Select records uniformly from each of the CSV files.
    # Place into the training list.
    select_done = False
    tfrecord_count = 0
    while not select_done:
        for fd in ordered_fd_list:
            line = fd.readline().strip()
            if (len(line) > 0):
                (pid, tfrtype, start, length) = line.split(',')
                tfstart = int(start)
                tflength = int(length)
                _, tfr_file = db.get_tfrecord_filename(pid, tfrtype, tfstart, tflength)
                train_list.append(tfr_file)
                tfrecord_count += 1
            else:
                # Out of records.
                select_done = True
        if (max_records > 0):
            if (tfrecord_count >= max_records):
                select_done = True
    
    # Close the master ordered CSV files.
    for fd in ordered_fd_list:
        fu.close_file(fd)
    
    # Create an array of random indices into the train array.
    rand_idx = rng.choice(tfrecord_count, size=tfrecord_count, replace=False)
    
    # Initialize test and validation lists.
    test_max = int(round(tfrecord_count * test_split / num_classes, 0))
    val_max = int(round(tfrecord_count * val_split / num_classes, 0))
    test_count = {}
    val_count = {}
    for rtype in rtype_list:
        test_count[rtype] = 0
        val_count[rtype] = 0
    
    # Create the test and validation lists.
    for idx in rand_idx:
        file = train_list[idx]
        for rtype in rtype_list:
            if ('_'+rtype+'_') in file:
                if (test_count[rtype] < test_max):
                    test_list.append(file)
                    test_count[rtype] += 1
                    break
                elif (val_count[rtype] < val_max):
                    val_list.append(file)
                    val_count[rtype] += 1
                    break

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
def ecg_map_2class(serialized):
    """
    Mapping function used to parse a tfrecord.
    
    Global variables ECG_LENGTH and ECG_FEATURE must be set.
    
    Returns a tuple of (ecg, one_hot), where one_hot is a 2-element vector
    corresponding to (N, AFIB).
    """
    global ECG_LENGTH
    global ECG_FEATURE
    features = tf.io.parse_single_example(
        serialized,
        features={
            'ecg_raw':   tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_fir':   tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_raw_z': tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_fir_z': tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'one_hot':   tf.io.FixedLenFeature([db.NUM_CLASSES], tf.float32),
    })
    ecg = features[ECG_FEATURE]

    # Convert one-hot to a 2-element vector.
    # Original: N = 1, AFIB = 3
    # Converted: N = 0, AFIB = 1
    one_hot = [0., 0.]
    max = tf.math.argmax(features['one_hot'])
    if (max == 1):
        one_hot[0] = 1.
    elif (max == 3):
        one_hot[1] = 1.
    
    return (ecg, one_hot)

# ----------------------------------------------------------------------
def ecg_map_4class(serialized):
    """
    Mapping function used to parse a tfrecord.
    
    Global variables ECG_LENGTH and ECG_FEATURE must be set.
    
    Returns a tuple of (ecg, one_hot), where one_hot is a 4-element vector
    corresponding to (Q, N, AFL, AFIB).
    """
    global ECG_LENGTH
    global ECG_FEATURE
    features = tf.io.parse_single_example(
        serialized,
        features={
            'ecg_raw':   tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_fir':   tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_raw_z': tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_fir_z': tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'one_hot':   tf.io.FixedLenFeature([db.NUM_CLASSES], tf.float32),
    })
    ecg = features[ECG_FEATURE]

    # Convert one-hot to a 4-element vector.
    # Original: Q = 0, N = 1, AFL = 2, AFIB = 3, J = 4
    # Remove the J position.
    one_hot = features['one_hot'][0:4]
    
    return (ecg, one_hot)
    
# ----------------------------------------------------------------------
def ecg_map_all(serialized):
    """
    Mapping function used to parse a tfrecord.
    
    Global variable ECG_LENGTH must be set.
    Global variable ECG_FEATURE is ignored.
    
    Returns a tuple of all ECG features and the one-hot vector:
    (ecg_raw, ecg_fir, ecg_raw_z, ecg_fir_z, one_hot)
    Used for development and debug; not suitable for training or inference.
    """
    global ECG_LENGTH
    features = tf.io.parse_single_example(
        serialized,
        features={
            'ecg_raw':   tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_fir':   tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_raw_z': tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'ecg_fir_z': tf.io.FixedLenFeature([ECG_LENGTH], tf.float32),
            'one_hot':   tf.io.FixedLenFeature([db.NUM_CLASSES], tf.float32),
    })
    ecg_raw   = features['ecg_raw']
    ecg_fir   = features['ecg_fir']
    ecg_raw_z = features['ecg_raw_z']
    ecg_fir_z = features['ecg_fir_z']
    one_hot   = features['one_hot']
    return (ecg_raw, ecg_fir, ecg_raw_z, ecg_fir_z, one_hot)


# ----------------------------------------------------------------------
# Classes.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Main program used as an example for test and debug.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print('mit_bih_afib_tfrecord test program not implemented.')
    
