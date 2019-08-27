import copy
import io
import os.path
import re
import struct

from .crc32c import crc32c


_VALID_OP_NAME_START = re.compile('^[A-Za-z0-9.]')
_VALID_OP_NAME_PART = re.compile('[A-Za-z0-9_.\\-/]+')

REGISTERED_FACTORIES = {}


def directory_check(path):
    """Initialize the directory for log files."""
    try:
        prefix = path.split(':')[0]
        factory = REGISTERED_FACTORIES[prefix]
        return factory.directory_check(path)
    except KeyError:
        if not os.path.exists(path):
            os.makedirs(path)


class RecordWriter(object):
    def __init__(self, path):
        self._name_to_tf_name = {}
        self._tf_names = set()
        self.path = path
        self._writer = open(path, 'wb')

    def write(self, data):
        w = self._writer.write
        header = struct.pack('Q', len(data))
        w(header)
        w(struct.pack('I', masked_crc32c(header)))
        w(data)
        w(struct.pack('I', masked_crc32c(data)))

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()


def masked_crc32c(data):
    x = u32(crc32c(data))
    return u32(((x >> 15) | u32(x << 17)) + 0xa282ead8)


def u32(x):
    return x & 0xffffffff


def make_valid_tf_name(name):
    if not _VALID_OP_NAME_START.match(name):
        # Must make it valid somehow, but don't want to remove stuff
        name = '.' + name
    return '_'.join(_VALID_OP_NAME_PART.findall(name))
