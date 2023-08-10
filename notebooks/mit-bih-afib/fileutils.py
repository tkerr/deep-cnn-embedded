# fileutils.py
# Convenience functions to perform file operations such as open and close.

# System packages.
import os
import sys
import tarfile
from pathlib import Path

# ----------------------------------------------------------------------
# Globals.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def open_file(file_name, mode='r'):
    """
    Convenience function to open a file and return a descriptor (or None if error)
    """
    fd = None
    try:
        fd = open(file_name, mode)
    except Exception as err:
        print('Error opening {}: {}'.format(file_name, str(err)))
    return fd

# ----------------------------------------------------------------------
def close_file(fd):
    """
    Convenience function to close a file and ignore errors.
    """
    try:
        fd.close()
    except Exception as err:
        print('Error closing file: {}'.format(str(err)))

# ----------------------------------------------------------------------
def create_file_from_list(filename, listname):
    """
    Create a file from a list of text items.  The items in the list are 
    added to the file line by line.
    """
    fd = open_file(filename, 'w')
    if fd is not None:
        for item in listname:
            fd.write('{}\n'.format(item))
        close_file(fd)

# ----------------------------------------------------------------------
def mkpath(path):
    """
    Convenience function to create a file path if it does not exist.
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as err:
        print('Error creating path {}: {}'.format(path, str(err)))

# ----------------------------------------------------------------------
def _tar_from_list(tf, filelist, progress, file_count):
    """
    Internal function to add files from a list to a tarfile.
    """
    for file in filelist:
        try:
            tf.add(file)
            file_count += 1
            if progress is not None:
                if ((file_count % progress) == 0):
                    print('{} '.format(file_count), end='')
        except Exception as err:
            print('Error: {}: {}'.format(file, str(err)))
    return file_count

# ----------------------------------------------------------------------
def mktar(filename, *listargs, progress=None):
    """
    Create an uncompressed tar file from one or more lists of input files.
    filename is the full name of the tarfile.
    listargs is a series of one or more lists of files to add to the archive.
    progress is an optional modulo number used to print a progress indicator.
    Returns the number of files added to the tarfile.
    Example: mktar('mytar.tar.gz', train_list, test_list, val_list)
    """
    file_count = 0
    tf = tarfile.open(filename, 'w:')
    for filelist in listargs:
        file_count = _tar_from_list(tf, filelist, progress, file_count)
    tf.close()
    if progress is not None:
        print(file_count)
    return file_count

# ----------------------------------------------------------------------
def addtar(filename, *listargs, progress=None):
    """
    Add files to an existing uncompressed tarfile from one or more lists of input files.
    filename is the full name of the tarfile.
    listargs is a series of one or more lists of files to add to the archive.
    progress is an optional modulo number used to print a progress indicator.
    Returns the number of files added to the tarfile.
    """
    file_count = 0
    tf = tarfile.open(filename, 'a:')
    for filelist in listargs:
        file_count = _tar_from_list(tf, filelist, progress, file_count)
    tf.close()
    if progress is not None:
        print(file_count)
    return file_count

