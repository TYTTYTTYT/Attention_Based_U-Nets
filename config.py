from os import path

# The home directory to store the Brain MRI dataset.
DATA_HOME = 'BrainMRI'

# The file path of the pickled numpy array dataset.
DATA_SET = path.join(f'{DATA_HOME}', 'data.bi')

# The invalid characters in the operating systems (Windows, Linux)
INVALID = {
    '/',
    '<',
    '>',
    '\\',
    ':',
    '"',
    '|',
    '?',
    '*'
    }