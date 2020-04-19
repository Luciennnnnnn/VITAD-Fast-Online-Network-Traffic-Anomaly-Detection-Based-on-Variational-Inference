import sys
sys.path.append("..")

from Algorithm.cores import *


if __name__ == '__main__':
    # run experiment for metrics tensor rank R
    start = time.time()
    eval_R('Abilene')
    end = time.time()
    print('Time Comsuption in Abilene on R:', end - start)

    start = time.time()
    eval_R('GEANT')
    end = time.time()
    print('Time Comsuption in GEANT on R:', end - start)

    start = time.time()
    eval_R('CERNET')
    end = time.time()
    print('Time Comsuption in CERNET on R:', end - start)