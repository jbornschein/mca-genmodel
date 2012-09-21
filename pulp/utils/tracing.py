#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: GPL V3.0
#

"""

Record tracepoint for runtime profiling/tracing.


Usage::
    import pulp.utils.tracing as tracing

    tracing.set_tracefile()

    tracing.tracepoint("Work::begin")
    # Do hard work
    tracing.tracepoint("Work::end")

"""

import os
import sys
import time
from mpi4py import MPI
from functools import wraps

trace_file = None
start_time = None


def tracepoint(str):
    """ Record a tracepoint *str*.

    If tracing was enables with *set_tracefile* the given tracepoint will be
    recorded.
    """
    global trace_file

    if trace_file is None:
        return

    global start_time
    ts = MPI.Wtime() - start_time
    trace_file.write("[%f] [%s]\n" % (ts, str))
    #f = sys._getframe()
    #c = f.f_code
    #trace_file.write("[%f] [%s] %s:%d %s\n" % (ts, str, c.co_filename, f.f_lineno, c.co_name))


def traced(func):
    """ Decorator for functions to be traced.

    Whenever a traced function is called two tracepoints will be recorded:
    "func_name:begin" when the function starts to execute and "func_name:end"
    when the function is about to return.

    Usage:
    
        @traced
        def some_function_or_method(...):
            # do something
    """
    begin_str = func.func_name + ':begin'
    end_str = func.func_name + ':end'
    @wraps(func)
    def wrapped(*args, **kwargs):
        tracepoint(begin_str)
        res = func(*args, **kwargs)
        tracepoint(end_str)
        return res
    return wrapped


def set_tracefile(fname="trace-%04d.txt", comm=MPI.COMM_WORLD):
    """ Enable tracing

    The fname argument is expected to have a %d format specifier which will
    be replaced with the MPI rank.
    """
    global trace_file
    global start_time

    fname = fname % comm.rank
    trace_file = open(fname, "w")
    trace_file.write("# Start time: %s\n" % time.asctime())
    trace_file.write("# Hostname: %s\n" % os.uname()[1])
    trace_file.write("# MPI size: %d rank: %d\n" % (comm.size, comm.rank))

    comm.Barrier(); start_time = MPI.Wtime()
    comm.Barrier(); start_time = MPI.Wtime()

