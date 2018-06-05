import sys, os
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.getcwd()))

Base_Path = './plot/npz'
Save_Path = './plot/fig'


def drawlineplot(npfile):
    name = npfile.split('.')[0]
    d = np.load(Base_Path+'/'+npfile)
    x = d['x']
    y = d['y']
    plt.plot(x, y)
    plt.xlabel('games')
    plt.ylabel('points')
    plt.title('points during training')
    plt.savefig(Save_Path+'/'+name+'.jpg')
    print 'saving '+name+' succeeded!'


def multidraw(np_list, name):
    x_list = []
    y_list = []
    for npf in np_list:
        d = np.load(Base_Path+'/'+npf)
        x_list.append(d['x'])
        y_list.append(d['y'])
    p_list = zip(x_list, y_list)

    plt.figure()
    for x, y in p_list:
        plt.plot(x, y)
    plt.xlabel('games')
    plt.ylabel('points')
    plt.title('points during training')
    plt.savefig(Save_Path+'/'+name+'.jpg')
    print 'saving '+name+' succeeded!'



if __name__ == "__main__":
    multidraw(['v100000.npz', 'random.npz'], 'contrast')
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

