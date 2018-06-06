import sys, os
import numpy as np
from plot import Base_Path
sys.path.append(os.path.dirname(os.getcwd()))


def evalAvg(np_list, start=None, end=None):
    avg = {}
    for npf in np_list:
        d = np.load(Base_Path+'/'+npf)
        print len(d['y'])
        if start and end:
            y = d['y'][start:end]
        elif start:
            y = d['y'][start]
        elif end:
            y = d['y'][:end]
        else:
            y = d['y']
        name = npf.split('.')[0]
        avg[name] = sum(y) / len(y)
    print avg

if __name__ == "__main__":
    evalAvg(['v1.npz', 'lr.npz', 'v10e6.npz'])
    exit(0)
    import inspect, sys
    current_module = sys.modules[__name__]
    funnamelst = [item[0] for item in inspect.getmembers(current_module, inspect.isfunction)]
    if len(sys.argv) > 1:
        index = 1
        while index < len(sys.argv):
            if '--' in sys.argv[index]:
   	            index += 2
            else:
                break
        func = getattr(sys.modules[__name__], sys.argv[index])
        func(*sys.argv[index+1:])
    else:
        print >> sys.stderr, '	'.join((__file__, "/".join(funnamelst), "args"))

