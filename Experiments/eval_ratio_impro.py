import sys
sys.path.append("..")

from Algorithm.cores2 import *
# call improved algorithm


if __name__ == '__main__':
    # run experiment for metrics outliers ratio
    start = time.time()
    eval_ratio('CERNET')
    end = time.time()
    print('Time Comsuption in CERNET on ratio:', end - start)