import sys
sys.path.append("..")

from Algorithm.cores import *


if __name__ == '__main__':
    # run experiment for metrics mu
    start = time.time()
    eval_time('Abilene')
    end = time.time()
    print('Time Comsuption in Abilene on time:', end - start)

    start = time.time()
    eval_time('GEANT')
    end = time.time()
    print('Time Comsuption in GEANT on time:', end - start)

    start = time.time()
    eval_time('CERNET')
    end = time.time()
    print('Time Comsuption in CERNET on time:', end - start)