"""
Read and write PCL .pcd files in python.
dimatura@cmu.edu, 2013-2018
- TODO better API for wacky operations.
- TODO add a cli for common operations.
- TODO deal properly with padding
- TODO deal properly with multicount fields
- TODO better support for rgb nonsense
"""

import re
import struct
import copy
from io import StringIO as sio
import numpy as np
import warnings
import lzf


numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_header(lines):
    """ Parse header of PCD files.
    """
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', str(ln))
        if not match:
            warnings.warn("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            #print('found size and count k %s  v %s '% (key, value))
            metadata[key] = list(map(int, value.split()))
            #print(list(map(int,value.split())))
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1]*len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def write_header(metadata, rename_padding=False):
    """ Given metadata as dictionary, return a string header.
    """
    template = """\
VERSION {version}
FIELDS {fields}
SIZE {size}
TYPE {type}
COUNT {count}
WIDTH {width}
HEIGHT {height}
VIEWPOINT {viewpoint}
POINTS {points}
DATA {data}
"""
    str_metadata = metadata.copy()

    if not rename_padding:
        str_metadata['fields'] = ' '.join(metadata['fields'])
    else:
        new_fields = []
        for f in metadata['fields']:
            if f == '_':
                new_fields.append('padding')
            else:
                new_fields.append(f)
        str_metadata['fields'] = ' '.join(new_fields)
    str_metadata['size'] = ' '.join(map(str, metadata['size']))
    str_metadata['type'] = ' '.join(metadata['type'])
    str_metadata['count'] = ' '.join(map(str, metadata['count']))
    str_metadata['width'] = str(metadata['width'])
    str_metadata['height'] = str(metadata['height'])
    str_metadata['viewpoint'] = ' '.join(map(str, metadata['viewpoint']))
    str_metadata['points'] = str(metadata['points'])
    tmpl = template.format(**str_metadata)
    return tmpl


def _metadata_is_consistent(metadata):
    """ Sanity check for metadata. Just some basic checks.
    """
    checks = []
    required = ('version', 'fields', 'size', 'width', 'height', 'points',
                'viewpoint', 'data')
    for f in required:
        if f not in metadata:
            print('%s required' % f)
    checks.append((lambda m: all([k in m for k in required]),
                   'missing field'))
    checks.append((lambda m: len(m['type']) == len(list(m['count'])) ==
                   len(m['fields']),
                   'length of type, count and fields must be equal'))
    checks.append((lambda m: m['height'] > 0,
                   'height must be greater than 0'))
    checks.append((lambda m: m['width'] > 0,
                   'width must be greater than 0'))
    checks.append((lambda m: m['points'] > 0,
                   'points must be greater than 0'))
    checks.append((lambda m: m['data'].lower() in ('ascii', 'binary',
                   'binary_compressed'),
                   'unknown data type:'
                   'should be ascii/binary/binary_compressed'))
    ok = True
    for check, msg in checks:
        if not check(metadata):
            print('error:', msg)
            ok = False
    return ok

# def pcd_type_to_numpy(pcd_type, pcd_sz):
#     """ convert from a pcd type string and size to numpy dtype."""
#     typedict = {'F' : { 4:np.float32, 8:np.float64 },
#                 'I' : { 1:np.int8, 2:np.int16, 4:np.int32, 8:np.int64 },
#                 'U' : { 1:np.uint8, 2:np.uint16, 4:np.uint32 , 8:np.uint64 }}
#     return typedict[pcd_type][pcd_sz]


def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.
    Note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.
    *TODO* allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type]*c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def build_ascii_fmtstr(pc):
    """ Make a format string for printing to ascii.
    Note %.8f is minimum for rgb.
    """
    fmtstr = []
    for t, cnt in zip(pc.type, pc.count):
        if t == 'F':
            fmtstr.extend(['%.10f']*cnt)
        elif t == 'I':
            fmtstr.extend(['%d']*cnt)
        elif t == 'U':
            fmtstr.extend(['%u']*cnt)
        else:
            raise ValueError("don't know about type %s" % t)
    return fmtstr


def parse_ascii_pc_data(f, dtype, metadata):
    """ Use numpy to parse ascii pointcloud data.
    """
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    rowstep = metadata['points']*dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    """ Parse lzf-compressed data.
    Format is undocumented but seems to be:
    - compressed size of data (uint32)
    - uncompressed size of data (uint32)
    - compressed data
    - junk
    """
    fmt = 'II'
    compressed_size, uncompressed_size =\
        struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    compressed_data = f.read(compressed_size)
    # TODO what to use as second argument? if buf is None
    # (compressed > uncompressed)
    # should we read buf as raw binary?
    buf = lzf.decompress(compressed_data, uncompressed_size)
    if len(buf) != uncompressed_size:
        raise IOError('Error decompressing data')
    # the data is stored field-by-field
    # JK: metadata['width'] replaced with metadata['width'] * metadata['height']
    pc_data = np.zeros(metadata['width'] * metadata['height'], dtype=dtype)
    ix = 0
    for dti in range(len(dtype)):
        dt = dtype[dti]
        bytes = dt.itemsize * metadata['width'] * metadata['height']
        column = np.fromstring(buf[ix:(ix+bytes)], dt)
        pc_data[dtype.names[dti]] = column
        ix += bytes
    return pc_data


def point_cloud_from_fileobj(f):
    """ Parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = f.readline().strip().decode('ascii')
        header.append(ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or\
                "binary_compressed"')
    return PointCloud(metadata, pc_data)


def point_cloud_from_path(fname):
    """ load point cloud in binary format
    """
    with open(fname, 'rb') as f:
        pc = point_cloud_from_fileobj(f)
    return pc


def point_cloud_from_buffer(buf):
    fileobj = sio.StringIO(buf)
    pc = point_cloud_from_fileobj(fileobj)
    fileobj.close()  # necessary?
    return pc


def update_field(pc, field, pc_data):
    """ Updates field in-place.
    """
    pc.pc_data[field] = pc_data
    return pc


def encode_rgb_for_pcl(rgb):
    """ Encode bit-packed RGB for use with PCL.
    :param rgb: Nx3 uint8 array with RGB values.
    :rtype: Nx1 float32 array with bit-packed RGB, for PCL.
    """
    assert(rgb.dtype == np.uint8)
    assert(rgb.ndim == 2)
    assert(rgb.shape[1] == 3)
    rgb = rgb.astype(np.uint32)
    rgb = np.array((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | (rgb[:, 2] << 0),
                   dtype=np.uint32)
    rgb.dtype = np.float32
    return rgb


def decode_rgb_from_pcl(rgb):
    """ Decode the bit-packed RGBs used by PCL.
    :param rgb: An Nx1 array.
    :rtype: Nx3 uint8 array with one column per color.
    """

    rgb = rgb.copy()
    rgb.dtype = np.uint32
    r = np.asarray((rgb >> 16) & 255, dtype=np.uint8)
    g = np.asarray((rgb >> 8) & 255, dtype=np.uint8)
    b = np.asarray(rgb & 255, dtype=np.uint8)
    rgb_arr = np.zeros((len(rgb), 3), dtype=np.uint8)
    rgb_arr[:, 0] = r
    rgb_arr[:, 1] = g
    rgb_arr[:, 2] = b
    return rgb_arr


class PointCloud(object):
    """ Wrapper for point cloud data.
    The variable members of this class parallel the ones used by
    the PCD metadata (and similar to PCL and ROS PointCloud2 messages),
    ``pc_data`` holds the actual data as a structured numpy array.
    The other relevant metadata variables are:
    - ``version``: Version, usually .7
    - ``fields``: Field names, e.g. ``['x', 'y' 'z']``.
    - ``size.`: Field sizes in bytes, e.g. ``[4, 4, 4]``.
    - ``count``: Counts per field e.g. ``[1, 1, 1]``. NB: Multi-count field
      support is sketchy.
    - ``width``: Number of points, for unstructured point clouds (assumed by
      most operations).
    - ``height``: 1 for unstructured point clouds (again, what we assume most
      of the time.
    - ``viewpoint``: A pose for the viewpoint of the cloud, as
      x y z qw qx qy qz, e.g. ``[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]``.
    - ``points``: Number of points.
    - ``type``: Data type of each field, e.g. ``[F, F, F]``.
    - ``data``: Data storage format. One of ``ascii``, ``binary`` or ``binary_compressed``.
    See `PCL docs <http://pointclouds.org/documentation/tutorials/pcd_file_format.php>`__
    for more information.
    """

    def __init__(self, metadata, pc_data):
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.pc_data = pc_data
        self.check_sanity()

    def get_metadata(self):
        """ returns copy of metadata """
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    def check_sanity(self):
        # pdb.set_trace()
        md = self.get_metadata()
        assert(_metadata_is_consistent(md))
        assert(len(self.pc_data) == self.points)
        assert(self.width*self.height == self.points)
        assert(len(self.fields) == len(self.count))
        assert(len(self.fields) == len(self.type))

    def copy(self):
        new_pc_data = np.copy(self.pc_data)
        new_metadata = self.get_metadata()
        return PointCloud(new_metadata, new_pc_data)

    @staticmethod
    def from_path(fname):
        return point_cloud_from_path(fname)

    @staticmethod
    def from_fileobj(fileobj):
        return point_cloud_from_fileobj(fileobj)

