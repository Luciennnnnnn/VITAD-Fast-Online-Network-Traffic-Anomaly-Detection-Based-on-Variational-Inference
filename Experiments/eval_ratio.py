import sys
sys.path.append("..")

from Algorithm.cores import *


if __name__ == '__main__':
    # run experiment for metrics outliers ratio
    start = time.time()
    eval_ratio('Abilene')
    end = time.time()
    print('Time Comsuption in Abilene on ratio:', end - start)

    start = time.time()
    eval_ratio('GEANT')
    end = time.time()
    print('Time Comsuption in GEANT on ratio:', end - start)

    start = time.time()
    eval_ratio('CERNET')
    end = time.time()
    print('Time Comsuption in CERNET on ratio:', end - start)