"""
The autotable module provides a simple interface to store data from simulation
runs into efficient HDF5 files onto the filesystem.

These files can later be opened 

 * in Python: PyTables (http://www.pytables.org/)
 * in Matlab: hdf5read( <filename>, <tablename> )
 * in pure C or C++: libhdf5 (http://www.hdfgroup.org/HDF5/)

Basic example::

    import numpy as np
    from pulp.utils import autotable

    tbl = autotable.AutoTable('~/testhdf.h5')
    for t in range(10):
        tbl.append('t', t)
        tbl.append('D', np.random.randn(20))

This code creates the file :file:`~/testhdf.h5` with two tables, each having 10
rows: The table *t* will store a single integer within each row, where the
table *D* will store a 20 element vector with gaussian distributed random 
numbers in each row.  
"""

import numpy as np
import tables

class AutoTable:
    """Store data into HDF5 files"""
    def __init__(self, fname=None, compression_level=1):
        """
        Create a new autotable object which will write data into a file called 
        fname. 

        If fname is not specified (or is None), fname will be derived from 
        sys.argv[0] by striping the extension and adding ".h5". As a result, the 
        file will be named like the creating program.

        Compression specifies the compression_level that should be applied when storing data.

        .. note:: If a file named fname existed previously, its content will be deleted!
        """
        if fname is None:
            fname = self._guess_fname()
        self.h5 = tables.openFile(fname, "w")
        self.compression_level = compression_level
        self.tables = {}
        self.types = {}

    def close(self):
        """
        Close the HDF file behind this AutoTable instance.
        """
        self.h5.close()

    def append(self, name, value):
        """
        Append the dataset *values* into a table called *name*. If a specified 
        table name does not exist, a a new table will be created. 

        Example:.

            tbl.append("T", temp)
            tbl.append("image", np.zeros((256,256)) )
        """
        if np.isscalar(value):
            value = np.asarray(value)

        if not isinstance(value, np.ndarray):
            raise TypeError("Don't know how to handle values of type '%s'", type(value))

        # Check if we need to create a new table
        if not self.tables.has_key(name):    
            self._create_table(name, value)

        value = value.reshape( (1,)+value.shape )
        try:
            self.tables[name].append(value)
        except ValueError:
            raise TypeError('Wrong datatype for "%s" field'%name)

        self.tables[name].flush()


    def append_all(self, valdict):
        """
        Append the given data to the table.

        *valdict* must be a dictionary containig key value pairs, where key
        is a string and specifies the table name. The corresponding value must be 
        an arbitrary numpy compatible value. If a specified table name does not
        exist, a a new table will be created. 

        Example::

            tbl.append( { 't':0.23 , 'D':np.zeros((10,10)) )
        """
        for name, value in valdict.items():
            self.append(name, value)

    def _create_table(self, name, example):
        """
        Create a new table within the HDF file, where the tables shape and its
        datatype are determined by *example*.
        """
        type_map = {
            np.dtype(np.float64) : tables.Float64Atom(),
            np.dtype(np.float32) : tables.Float32Atom(),
            np.dtype(np.int)     : tables.Int64Atom(),
            np.dtype(np.int8)    : tables.Int8Atom(),
            np.dtype(np.uint8)   : tables.UInt8Atom(),
            np.dtype(np.int16)   : tables.Int16Atom(),
            np.dtype(np.uint16)  : tables.UInt16Atom(),
            np.dtype(np.int32)   : tables.Int32Atom(),
            np.dtype(np.uint32)  : tables.UInt32Atom(),
        }

        try:
            h5type = type_map[example.dtype]
        except KeyError:
            raise TypeError("Don't know how to handle dtype '%s'" % example.dtype)

        h5dim = (0,) + example.shape

        h5 = self.h5
        filters = tables.Filters(complevel=self.compression_level, complib='zlib', shuffle=True)
        self.tables[name] = h5.createEArray( h5.root, name, h5type, h5dim, filters=filters )
        self.types[name] = type(example)

    def _guess_fname(self):
        """
        Derive an fname from sys.argv[0] by striping the extension and adding ".h5". 
        As a result, the table will be named just like the executing programm.
        """
        import sys
        import os.path as path
    
        base, _ = path.splitext(sys.argv[0])
        return base+".h5"

