import re
import os


def natsorted(a_list):
    '''
    Return a_list naturally sorted. See natural sorting.
    '''
    def convert(e): return int(e) if e.isdigit() else e.lower()

    def key(e): return [convert(c) for c in re.split('([0-9]+)', e)]
    return sorted(a_list, key=key)


def cleanup(fname):
    try:
        os.remove(fname)
    except OSError:
        pass
