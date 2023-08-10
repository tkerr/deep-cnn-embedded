# tflite_predict.py
# Run a set of predictions on a TensorFlow Lite model.

# System packages.
import os
import pickle
import platform
import time
import sys

# TensorFlow Lite is platform-specific
if (platform.machine() == 'aarch64'):
    import tflite_runtime.interpreter as tfl  # Raspberry Pi
else:
    import tensorflow.lite as tfl  # Others


# ----------------------------------------------------------------------
# Globals.
# ----------------------------------------------------------------------
DATASET_PATH_ROOT = os.path.abspath(r'E:/Data/MIT-BIH-AFIB')
LOCAL_DATA_PATH = os.path.join(DATASET_PATH_ROOT, 'pkl')


# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def load_model(model_file_name):
    """
    Load a TensorFlow Lite model interpreter from a file.
    Returns the interpreter.
    """
    interp = tfl.Interpreter(model_file_name)
    interp.allocate_tensors()
    return interp

# ----------------------------------------------------------------------
def run_predictions(master_list_file, interp, data_path=LOCAL_DATA_PATH):
    """
    Run predictions on a set of data files.
    master_list_file is a CSV file containing a list of data files.
    interp is the TensorFlow Lite interpreter.
    Returns a predictions array.
    """
    elapsed_time = 0.0
    pred_list = []
    input = interp.tensor(interp.get_input_details()[0]['index'])
    output_index = interp.get_output_details()[0]['index']
    
    list_fd = open(master_list_file, 'r')
    for line in list_fd:
        data_file = os.path.join(data_path, line.strip().split(',')[0])
        with open(data_file, 'rb') as data_fd:
            raw_data = pickle.load(data_fd)
        start = time.time()
        input()[0,:,0] = raw_data  # expected shape is (1,7500,1)
        interp.invoke()
        output_data = interp.get_tensor(output_index)
        delta_time = time.time() - start
        elapsed_time += delta_time
        pred_list.append(list(output_data))
    data_fd.close()
    print('{} predictions completed.'.format(len(pred_list)))
    print('Avg time: {:0.6f} s'.format(elapsed_time / len(pred_list)))
    return pred_list


# ----------------------------------------------------------------------
# Main program.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    """
    Run a set of predictions on a TensorFlow Lite model.
    model_file is the TensorFlow Lite model file name.
    master_file_list is a CSV file containing a list of data files to run.
    """
    if (len(sys.argv) < 3):
        print('Usage: tflite_predict.py model_file master_file_list')
        sys.exit(1)
        
    interp = load_model(sys.argv[1])
    master_list_file = os.path.join(LOCAL_DATA_PATH, sys.argv[2])
    predictions = run_predictions(master_list_file, interp)
    predictions_file = os.path.abspath('predictions.pkl')
    with open(predictions_file, 'wb') as fd:
        pickle.dump(predictions, fd)
    print('Predictions written to {}'.format(predictions_file))
    