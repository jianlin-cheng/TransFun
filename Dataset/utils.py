import os
import re
import subprocess
from pathlib import Path


patterns = {
    'pdb': r'pdb[0-9]*$',
    'pdb.gz': r'pdb[0-9]*\.gz$',
    'mmcif': r'(mm)?cif$',
    'sdf': r'sdf[0-9]*$',
    'xyz': r'xyz[0-9]*$',
    'xyz-gdb': r'xyz[0-9]*$',
    'silent': r'out$',
    'sharded': r'@[0-9]+',
}

_regexes = {k: re.compile(v) for k, v in patterns.items()}


def is_type(f, filetype):
    if filetype in _regexes:
        return _regexes[filetype].search(str(f))
    else:
        return re.compile(filetype + r'$').search(str(f))


def find_files(path, suffix, relative=None):
    """
    Find all files in path with given suffix. =

    :param path: Directory in which to find files.
    :type path: Union[str, Path]
    :param suffix: Suffix determining file type to search for.
    :type suffix: str
    :param relative: Flag to indicate whether to return absolute or relative path.

    :return: list of paths to all files with suffix sorted by their names.
    :rtype: list[str]
    """
    if not relative:
        find_cmd = r"find {:} -regex '.*\.{:}' | sort".format(path, suffix)
    else:
        find_cmd = r"cd {:}; find . -regex '.*\.{:}' | cut -d '/' -f 2- | sort" \
            .format(path, suffix)
    out = subprocess.Popen(
        find_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.getcwd(), shell=True)
    (stdout, stderr) = out.communicate()
    name_list = stdout.decode().split()
    name_list.sort()
    return sorted([Path(x) for x in name_list])