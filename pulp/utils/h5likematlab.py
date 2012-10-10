#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

from autotable import AutoTable
import tables

def hdf5write(path, name, variable):
    file_ = AutoTable(path)
    file_.append(name, variable)
    file_.close()
    
def hdf5read(path, name, variable):
    if name[-2:] != 'h5':
        name += '.h5'
    file_ = tables.openFile(path + name)
    exec('output = file_.root.' + variable + '[:]')
    file_.close()
    return output

def hdf5info(path, name):
    if name[-2:] != 'h5':
        name += '.h5'
    file_ = tables.openFile(path + name)
    output = []
    for item in file_.root:
        output.append(item.name)
    file_.close()
    return output