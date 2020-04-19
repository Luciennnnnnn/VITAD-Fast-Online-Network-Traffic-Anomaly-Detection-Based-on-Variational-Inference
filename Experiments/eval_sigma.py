import sys
sys.path.append("..")

from Algorithm.cores import *


if __name__ == '__main__':
    # run experiment for metrics outliers ratio
    start = time.time()
    eval_sigma('Abilene')
    end = time.time()
    print('Time Comsuption in Abilene on sigma:', end - start)

    start = time.time()
    eval_sigma('GEANT')
    end = time.time()
    print('Time Comsuption in GEANT on sigma:', end - start)

    start = time.time()
    eval_sigma('CERNET')
    end = time.time()
    print('Time Comsuption in CERNET on sigma:', end - start)