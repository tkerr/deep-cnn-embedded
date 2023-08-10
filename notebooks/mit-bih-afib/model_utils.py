# model_utils.py
# Utility functions to save and load TensorFlow neural net models, weights and results.

import os
import json
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json


# ----------------------------------------------------------------------
# Globals.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------

def save_model(tf_model, model_file_base):
    """
    Save a model in TensorFlow (.tf) and Keras H5 (.h5) formats, and
    save a JSON architecture string.
    
    Parameters:
        tf_model : The TensorFlow model to save
        model_file_base : The path and base name of the model file, without
            any file extension; e.g., path/to/my_model. File extensions are
            added to the base name.
    
    Returns:
        True if successful, False otherwise.
    """
    saved = False
    try:
        tf_model.save(model_file_base, save_format='tf')          # Save in TensorFlow SavedModel format
        tf_model.save(model_file_base + '.h5', save_format='h5')  # Save in Keras H5 format
        json_config = tf_model.to_json()                          # Save a JSON architecture string
        with open(model_file_base + '.json', 'w') as fd:
            fd.write(json_config)
        fd.close()
        saved = True
    except Exception as err:
        print('Error saving {}: {}'.format(model_file_base, str(err)))
    return saved

# ----------------------------------------------------------------------
def load_model(model_file_spec):
    """
    Load a TensorFlow model from a file. 
    
    Parameters:
        model_file_spec : The model file path or filename to load.
            If a directory, then a TensorFlow SavedModel format is assumed.  
            If a file, then it must have a .h5 or .json extension.

    Returns:
        The TensorFlow model, or None if an error occurred.
    """
    tf_model = None
    
    if os.path.isdir(model_file_spec):
        try:
            tf_model = tf.keras.models.load_model(model_file_spec)
        except Exception as err:
            print('Error loading {}: {}'.format(model_file_spec, str(err)))
    else:
        _, ext = os.path.splitext(model_file_spec)
    
        if (ext == '.h5'):
            try:
                tf_model = tf.keras.models.load_model(model_file_spec)
            except Exception as err:
                print('Error loading {}: {}'.format(model_file_spec, str(err)))
        
        elif (ext == '.json'):
            try:
                with open(model_file_spec, 'r') as fd:
                    json_config = json.load(fd)
                fd.close()
                json_str = json.dumps(json_config)
                tf_model = model_from_json(json_str)
            except Exception as err:
                print('Error loading {}: {}'.format(model_file_spec, str(err)))
        else:
            print('Unknown model file format: {}'.format(model_file_spec))
    
    return tf_model

# ----------------------------------------------------------------------
def save_weights(tf_model, model_file_base):
    """
    Save TensorFlow model weights in both TensorFlow (.tf) and Keras H5 (.h5) formats.
    
    Parameters:
        tf_model : The TensorFlow model to save weights from
        model_file_base : The path and base name of the model file, without
            any file extension; e.g., path/to/my_model. File extensions are
            added to the base name.
    
    Returns:
        True if successful, False otherwise.
    """
    saved = False
    try:
        tf_model.save_weights(model_file_base, save_format='tf')          # Save in TensorFlow SavedModel format
        tf_model.save_weights(model_file_base + '.h5', save_format='h5')  # Save in Keras H5 format
        saved = True
    except Exception as err:
        print('Error saving {}: {}'.format(model_file_base, str(err)))
    return saved

# ----------------------------------------------------------------------
def load_weights(tf_model, weights_path):
    """
    Load TensorFlow model weights from a file or file path. 
    
    Parameters:
        tf_model : The TensorFlow model to load weights into
        weights_path : The file path or name.  If a directory, then
            a TensorFlow SavedModel format is assumed.  If a file,
            then a Keras H5 format is assumed.            
    
    Returns:
        True if successful, False otherwise.
    """
    loaded = False
    by_name = False if os.path.isdir(weights_path) else True
    try:
        tf_model.load_weights(weights_path, by_name=False)
        loaded = True
    except Exception as err:
        print('Error loading {}: {}'.format(weights_path, str(err)))
    return loaded

# ----------------------------------------------------------------------
def save_results(results, results_file):
    """
    Save a training results dictionary to a pickle file.
    
    Parameters:
        results : The results dictionary to save
        results_file : The name of the results file to save.
    
    Returns:
        True if successful, False otherwise.
    """
    saved = False
    try:
        with open(results_file, 'wb') as fd:
            pickle.dump(results, fd)
        fd.close()
        saved = True
    except Exception as err:
        print('Error saving {}: {}'.format(results_file, str(err)))
    return saved

# ----------------------------------------------------------------------
def load_results(results_file):
    """
    Load a training results dictionary from a pickle file.
    
    Parameters:
        results_file : The name of the results file to load.
    
    Returns:
        The results dictionary if successful, or an empty dictionary if an error occurred.
    """
    results = {}
    try:
        with open(results_file, 'rb') as fd:
            results = pickle.load(fd)
        fd.close()
    except Exception as err:
        print('Error loading {}: {}'.format(results_file, str(err)))
    return results

