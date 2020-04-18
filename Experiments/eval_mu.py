from proposed import *
from cores import *

if __name__ == '__main__':
    # run experiment for metrics mu
    start = time.time()
    eval_mu('Abilene')
    end = time.time()
    print('Time Comsuption in Abilene on mu:', end - start)

    start = time.time()
    eval_mu('GEANT')
    end = time.time()
    print('Time Comsuption in GEANT on mu:', end - start)

    start = time.time()
    eval_mu('CERNET')
    end = time.time()
    print('Time Comsuption in CERNET on mu:', end - start)