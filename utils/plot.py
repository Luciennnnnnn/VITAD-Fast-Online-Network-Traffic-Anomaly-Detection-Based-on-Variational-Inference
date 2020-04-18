import matplotlib
import numpy as np
import os
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
matplotlib.use('Agg')


def plot_R(dataset_name):
    x = range(2, 51, 2)
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, os.path.join('proposed', 'R_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())

    plt.figure(figsize=(2.5, 1.6))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')

    plt.xticks(range(0, len(x), 4), range(2, 51, 8), fontsize=14)
    plt.yticks(fontsize=12)
    plt.xlabel('rank (R)', fontsize=12)
    plt.ylabel('TPR', fontsize=12)

    #plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #plt.margins(0, 0)

    plt.savefig(os.path.join(pre, 'R_TPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'R_TPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join(pre, 'R_TPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'R_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())

    plt.figure(figsize=(2.5, 1.6))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')

    plt.xticks(range(0, len(x), 4), range(2, 51, 8), fontsize=14)
    plt.yticks(fontsize=12)
    plt.xlabel('rank (R)', fontsize=12)
    plt.ylabel('FPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'R_FPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'R_FPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join(pre, 'R_FPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.close()


def plot_mu(dataset_name):
    x = [0.01, 0.05, 0.1, 0.5, 1]
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, os.path.join('proposed', 'mu_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'mu_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'mu_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'mu_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'mu_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'mu_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'mu_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    #plt.figure(figsize=(2, 1.5))
    plt.subplot(2, 3, 1)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(0.84, 0.825), loc=2, borderaxespad=0, ncol=4)
    plt.xlabel('mean value $\\mu$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'mu_TPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'mu_TPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join(pre, 'mu_TPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'mu_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'mu_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'mu_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'mu_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'mu_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'mu_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'mu_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.figure(figsize=(11.5, 6.6))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('mean value $\\mu$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'mu_FPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'mu_FPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join(pre, 'mu_FPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.close()


def plot_sigma(dataset_name):
    x = [0.01, 0.05, 0.1, 0.5, 1]
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, os.path.join('proposed', 'sigma_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'sigma_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'sigma_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'sigma_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'sigma_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'sigma_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'sigma_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.figure(figsize=(11.5, 6.6))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('variance $\\sigma^2$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'sigma_TPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'sigma_TPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join(pre, 'sigma_TPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'sigma_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'sigma_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'sigma_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'sigma_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'sigma_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'sigma_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'sigma_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.figure(figsize=(11.5, 6.6))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('variance $\\sigma^2$')
    plt.ylabel('FPR')
    plt.savefig(os.path.join(pre, 'sigma_FPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'sigma_FPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join(pre, 'sigma_FPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.close()


def plot_ratio(dataset_name):
    x = np.arange(1, 11)
    x = x / 100
    pre = os.path.join(os.path.join('results', dataset_name))

    with open(os.path.join(pre, os.path.join('proposed', 'ratio_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'ratio_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'ratio_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'ratio_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'ratio_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'ratio_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'ratio_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.figure(figsize=(11.5, 6.6))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('outlier ratio $\\gamma$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'ratio_TPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'ratio_TPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join(pre, 'ratio_TPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'ratio_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'ratio_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'ratio_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'ratio_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'ratio_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'ratio_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'ratio_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.figure(figsize=(11.5, 6.6))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('outlier ratio $\\gamma$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'ratio_FPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'ratio_FPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join(pre, 'ratio_FPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.close()


def plot_time(dataset_name):
    x = range(0, 8000, 100)
    pre = os.path.join(os.path.join('results', dataset_name))
    tmp = [_ for _ in range(0, 80, 8)]
    tmp.append(79)
    tmp2 = [x[_]+1 for _ in range(0, 80, 8)]
    tmp2.append(8000)
    with open(os.path.join(pre, os.path.join('proposed', 'TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())

    y1 = [y1[_] for _ in range(0, 8000, 100)]
    y2 = [y2[_] for _ in range(0, 8000, 100)]
    y3 = [y3[_] for _ in range(0, 8000, 100)]
    y4 = [y4[_] for _ in range(0, 8000, 100)]
    y5 = [y5[_] for _ in range(0, 8000, 100)]
    y6 = [y6[_] for _ in range(0, 8000, 100)]
    print(len(y1))
    plt.figure(figsize=(10.5, 6.5))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#faaf3a', marker='<', linestyle=':', label='T-online')

    plt.xticks(tmp, tmp2, fontsize=12)
    plt.legend(bbox_to_anchor=(0.056, 0.454), loc=2, borderaxespad=0)
    #plt.ylim(ymin=0, ymax=1.0)
    plt.xlabel('sample point t', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'TPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.savefig(os.path.join(pre, 'TPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'TPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())

    y1 = [y1[_] for _ in range(0, 8000, 100)]
    y2 = [y2[_] for _ in range(0, 8000, 100)]

    y3 = [y3[_] for _ in range(0, 8000, 100)]
    y4 = [y4[_] for _ in range(0, 8000, 100)]
    y5 = [y5[_] for _ in range(0, 8000, 100)]
    y6 = [y6[_] for _ in range(0, 8000, 100)]
    print(len(y1))
    plt.figure(figsize=(10.5, 6.5))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(tmp, tmp2, fontsize=12)
    plt.legend(bbox_to_anchor=(0.825, 0.985), loc=2, borderaxespad=0)
    #plt.ylim(ymin=0, ymax=1.0)
    plt.xlabel('sample point t', fontsize=12)
    plt.ylabel('FPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'FPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'FPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join(pre, 'FPRS.png'), bbox_inches='tight', dpi=300, format='png')
    # plt.clf()
    #
    # with open(os.path.join(pre, os.path.join('proposed', 'RSE.json')), 'r') as FR:
    #     y1 = json.loads(FR.read())
    # with open(os.path.join(pre, os.path.join('OSTD', 'RSE.json')), 'r') as FR:
    #     y2 = json.loads(FR.read())
    # with open(os.path.join(pre, os.path.join('STOC-RPCA', 'RSE.json')), 'r') as FR:
    #     y3 = json.loads(FR.read())
    # with open(os.path.join(pre, os.path.join('ReProCS', 'RSE.json')), 'r') as FR:
    #     y4 = json.loads(FR.read())
    # with open(os.path.join(pre, os.path.join('FBCP', 'RSE.json')), 'r') as FR:
    #     y5 = json.loads(FR.read())
    # with open(os.path.join(pre, os.path.join('T-online', 'RSE.json')), 'r') as FR:
    #     y6 = json.loads(FR.read())
    #
    # y1 = [y1[_] for _ in range(0, 8000, 100)]
    # y2 = [y2[_] for _ in range(0, 8000, 100)]
    # y3 = [y3[_] for _ in range(0, 8000, 100)]
    # y4 = [y4[_] for _ in range(0, 8000, 100)]
    # y5 = [y5[_] for _ in range(0, 8000, 100)]
    # print(len(y1))
    # plt.figure(figsize=(10.5, 6.5))
    # plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    # plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    # plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    # plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    # plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    # plt.plot(range(len(x)), y6, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    # plt.xticks(tmp, tmp2, fontsize=12)
    # plt.legend(bbox_to_anchor=(0.89, 0.98), loc=2, borderaxespad=0)
    # #plt.ylim(ymin=0, ymax=1.0)
    # plt.xlabel('sample point t', fontsize=12)
    # plt.ylabel('RSE', fontsize=12)
    # plt.savefig(os.path.join(pre, 'RSE.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    # plt.savefig(os.path.join(pre, 'RSE.eps'), bbox_inches='tight', dpi=300, format='eps')
    # plt.savefig(os.path.join(pre, 'RSE.png'), bbox_inches='tight', dpi=300, format='png')
    plt.close()


def plot_mu_n():
    x = [0.01, 0.05, 0.1, 0.5, 1]
    pre = os.path.join(os.path.join('results', 'Abilene'))

    with open(os.path.join(pre, os.path.join('proposed', 'mu_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'mu_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'mu_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'mu_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'mu_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'mu_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'mu_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.figure(figsize=(10.5, 6.5))
    plt.subplot(2, 3, 1)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(0.84, 0.825), loc=2, borderaxespad=0, ncol=4)
    plt.xlabel('mean value $\\mu$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'GEANT'))
    with open(os.path.join(pre, os.path.join('proposed', 'mu_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'mu_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'mu_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'mu_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'mu_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'mu_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'mu_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 2)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('mean value $\\mu$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'mu_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'mu_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'mu_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'mu_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'mu_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'mu_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'mu_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 3)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('mean value $\\mu$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'Abilene'))
    with open(os.path.join(pre, os.path.join('proposed', 'mu_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'mu_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'mu_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'mu_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'mu_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'mu_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'mu_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 4)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('mean value $\\mu$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(1, -0.224, '(a) Abilene', fontsize=12, weight='bold')

    pre = os.path.join(os.path.join('results', 'GEANT'))
    with open(os.path.join(pre, os.path.join('proposed', 'mu_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'mu_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'mu_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'mu_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'mu_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'mu_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'mu_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 5)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('mean value $\\mu$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(1, -0.047, '(b) GEANT', fontsize=12, weight='bold')

    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'mu_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'mu_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'mu_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'mu_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'mu_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'mu_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'mu_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'mu_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 6)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(-2.664, 2.39), loc=2, borderaxespad=0, ncol=8, handletextpad=0.6, columnspacing=1)
    plt.xlabel('mean value $\\mu$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(1, -0.205, '(c) CERNET', fontsize=12, weight='bold')
    plt.subplots_adjust(wspace=0.33, hspace=0.24)
    plt.savefig(os.path.join('results', 'mu.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join('results', 'mu.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join('results', 'mu.png'), bbox_inches='tight', dpi=300, format='png')
    plt.close()


def plot_sigma_n():
    x = [0.01, 0.05, 0.1, 0.5, 1]
    pre = os.path.join(os.path.join('results', 'Abilene'))

    with open(os.path.join(pre, os.path.join('proposed', 'sigma_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'sigma_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'sigma_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'sigma_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'sigma_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'sigma_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'sigma_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'sigma_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.figure(figsize=(10.5, 6.5))
    plt.subplot(2, 3, 1)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(0.84, 0.825), loc=2, borderaxespad=0, ncol=4)
    plt.xlabel('variance $\\sigma^2$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'GEANT'))
    with open(os.path.join(pre, os.path.join('proposed', 'sigma_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'sigma_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'sigma_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'sigma_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'sigma_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'sigma_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'sigma_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'sigma_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 2)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('variance $\\sigma^2$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'sigma_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'sigma_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'sigma_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'sigma_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'sigma_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'sigma_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'sigma_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'sigma_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 3)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('variance $\\sigma^2$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'Abilene'))
    with open(os.path.join(pre, os.path.join('proposed', 'sigma_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'sigma_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'sigma_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'sigma_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'sigma_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'sigma_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'sigma_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'sigma_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 4)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('variance $\\sigma^2$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(1, -0.2347, '(a) Abilene', fontsize=12, weight='bold')

    pre = os.path.join(os.path.join('results', 'GEANT'))
    with open(os.path.join(pre, os.path.join('proposed', 'sigma_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'sigma_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'sigma_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'sigma_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'sigma_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'sigma_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'sigma_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'sigma_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 5)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('variance $\\sigma^2$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(1, -0.226, '(b) GEANT', fontsize=12, weight='bold')

    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'sigma_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'sigma_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'sigma_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'sigma_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'sigma_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'sigma_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'sigma_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'sigma_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 6)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(-2.626, 2.37), loc=2, borderaxespad=0, ncol=8, handletextpad=0.6, columnspacing=1)
    plt.xlabel('variance $\\sigma^2$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(1, -0.198, '(c) CERNET', fontsize=12, weight='bold')
    plt.subplots_adjust(wspace=0.31, hspace=0.22)
    plt.savefig(os.path.join('results', 'sigma.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join('results', 'sigma.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join('results', 'sigma.png'), bbox_inches='tight', dpi=300, format='png')
    plt.close()


def plot_ratio_n():
    x = np.arange(1, 11)
    x = x / 100
    pre = os.path.join(os.path.join('results', 'Abilene'))

    with open(os.path.join(pre, os.path.join('proposed', 'ratio_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'ratio_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'ratio_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'ratio_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'ratio_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'ratio_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'ratio_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'ratio_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.figure(figsize=(10.5, 6.5))
    plt.subplot(2, 3, 1)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(0, len(x), 2), [x[_] for _ in range(0, len(x), 2)], fontsize=12)
    #plt.legend(bbox_to_anchor=(0.84, 0.825), loc=2, borderaxespad=0, ncol=4)
    plt.xlabel('outlier ratio $\\gamma$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'GEANT'))
    with open(os.path.join(pre, os.path.join('proposed', 'ratio_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'ratio_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'ratio_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'ratio_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'ratio_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'ratio_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'ratio_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'ratio_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 2)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(0, len(x), 2), [x[_] for _ in range(0, len(x), 2)], fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('outlier ratio $\\gamma$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'ratio_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'ratio_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'ratio_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'ratio_FPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'ratio_FPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'ratio_FPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'ratio_FPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'ratio_FPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 3)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(0, len(x), 2), [x[_] for _ in range(0, len(x), 2)], fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('outlier ratio $\\gamma$', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'Abilene'))
    with open(os.path.join(pre, os.path.join('proposed', 'ratio_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'ratio_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'ratio_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'ratio_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'ratio_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'ratio_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'ratio_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'ratio_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 4)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(0, len(x), 2), [x[_] for _ in range(0, len(x), 2)], fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('outlier ratio $\\gamma$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(2.16, -0.322, '(a) Abilene', fontsize=12, weight='bold')

    pre = os.path.join(os.path.join('results', 'GEANT'))
    with open(os.path.join(pre, os.path.join('proposed', 'ratio_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'ratio_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'ratio_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'ratio_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'ratio_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'ratio_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'ratio_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'ratio_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 5)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(0, len(x), 2), [x[_] for _ in range(0, len(x), 2)], fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)outlier ratio $\\gamma$outlier ratio $\\gamma$outlier ratio $\\gamma$outlier ratio $\\gamma$outlier ratio $\\gamma$outlier ratio $\\gamma$outlier ratio $\\gamma$outlier ratio $\\gamma$outlier ratio $\\gamma$outlier ratio $\\gamma$outlier ratio $\\gamma$
    plt.xlabel('outlier ratio $\\gamma$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(2.16, -0.1235, '(b) GEANT', fontsize=12, weight='bold')

    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'ratio_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('OSTD', 'ratio_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('STOC-RPCA', 'ratio_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('ReProCS', 'ratio_TPRS.json')), 'r') as FR:
        y4 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('FBCP', 'ratio_TPRS.json')), 'r') as FR:
        y5 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('3WD', 'ratio_TPRS.json')), 'r') as FR:
        y6 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('CP-ALS', 'ratio_TPRS.json')), 'r') as FR:
        y7 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'ratio_TPRS.json')), 'r') as FR:
        y8 = json.loads(FR.read())

    plt.subplot(2, 3, 6)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='OSTD')
    plt.plot(range(len(x)), y3, color='#ddccc5', marker='D', linestyle='-.', label='STOC-RPCA')
    plt.plot(range(len(x)), y4, color='#ffa289', marker='h', linestyle=':', label='ReProCS')
    plt.plot(range(len(x)), y5, color='#e24b2c', marker='v', linestyle=':', label='FBCP')
    plt.plot(range(len(x)), y6, color='#6e7a8a', marker='^', linestyle=':', label='3WD')
    plt.plot(range(len(x)), y7, color='#c1194d', marker='<', linestyle=':', label='CP-ALS')
    plt.plot(range(len(x)), y8, color='#faaf3a', marker='<', linestyle=':', label='T-online')
    plt.xticks(range(0, len(x), 2), [x[_] for _ in range(0, len(x), 2)], fontsize=12)
    plt.legend(bbox_to_anchor=(-2.644, 2.369), loc=2, borderaxespad=0, ncol=8, handletextpad=0.6, columnspacing=1)
    plt.xlabel('outlier ratio $\\gamma$', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(2.16, -0.270, '(c) CERNET', fontsize=12, weight='bold')
    plt.subplots_adjust(wspace=0.32, hspace=0.22)
    plt.savefig(os.path.join('results', 'ratio.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join('results', 'ratio.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join('results', 'ratio.png'), bbox_inches='tight', dpi=300, format='png')
    plt.close()


def plot_R_n():
    x = range(2, 51, 2)
    pre = os.path.join(os.path.join('results', 'Abilene'))

    with open(os.path.join(pre, os.path.join('proposed', 'R_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())

    plt.figure(figsize=(10.5, 6.5))
    plt.subplot(2, 3, 1)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.xticks(range(0, len(x), 4), range(2, 51, 8), fontsize=12)
    #plt.legend(bbox_to_anchor=(0.84, 0.825), loc=2, borderaxespad=0, ncol=4)
    plt.xlabel('rank (R)', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'GEANT'))
    with open(os.path.join(pre, os.path.join('proposed', 'R_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())

    plt.subplot(2, 3, 2)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.xticks(range(0, len(x), 4), range(2, 51, 8), fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('rank (R)', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'R_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())

    plt.subplot(2, 3, 3)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.xticks(range(0, len(x), 4), range(2, 51, 8), fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('rank (R)', fontsize=12)
    plt.ylabel('FPR', fontsize=12)

    pre = os.path.join(os.path.join('results', 'Abilene'))
    with open(os.path.join(pre, os.path.join('proposed', 'R_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())

    plt.subplot(2, 3, 4)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.xticks(range(0, len(x), 4), range(2, 51, 8), fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('rank (R)', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(4.83, 0.99867, '(a) Abilene', fontsize=12, weight='bold')

    pre = os.path.join(os.path.join('results', 'GEANT'))
    with open(os.path.join(pre, os.path.join('proposed', 'R_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())

    plt.subplot(2, 3, 5)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.xticks(range(0, len(x), 4), range(2, 51, 8), fontsize=12)
    #plt.legend(bbox_to_anchor=(1.0, 1), loc=2, borderaxespad=0)
    plt.xlabel('rank (R)', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(5.25, 0.945966, '(b) GEANT', fontsize=12, weight='bold')

    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'R_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())

    plt.subplot(2, 3, 6)
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.xticks(range(0, len(x), 4), range(2, 51, 8), fontsize=12)
    plt.xlabel('rank (R)', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.text(4.9, 0.799192, '(c) CERNET', fontsize=12, weight='bold')
    plt.subplots_adjust(wspace=0.52, hspace=0.22)
    plt.savefig(os.path.join('results', 'R.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join('results', 'R.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join('results', 'R.png'), bbox_inches='tight', dpi=300, format='png')
    plt.close()


def plot_SNR_n():
    x = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'SNR_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'SNR_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())

    plt.figure(figsize=(10.5, 6.5))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='T-online')

    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(0.056, 0.454), loc=2, borderaxespad=0)
    # plt.ylim(ymin=0, ymax=1.0)
    plt.xlabel('sample point t', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'SNR_TPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.savefig(os.path.join(pre, 'SNR_TPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'SNR_TPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'SNR_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'SNR_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())

    plt.figure(figsize=(10.5, 6.5))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='proposed')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='T-online')

    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(0.056, 0.454), loc=2, borderaxespad=0)
    # plt.ylim(ymin=0, ymax=1.0)
    plt.xlabel('sample point t', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'SNR_FPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.savefig(os.path.join(pre, 'SNR_FPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'SNR_FPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.close()


def plot_ratio_n2():
    x = np.arange(1, 11)
    x = x / 100
    pre = os.path.join(os.path.join('results', 'CERNET'))
    with open(os.path.join(pre, os.path.join('proposed', 'ratio_original_TPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'ratio_TPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('proposed', 'ratio_n_TPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())

    plt.figure(figsize=(10.5, 6.5))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='original')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='T-online')
    plt.plot(range(len(x)), y3, color='#e56a6c', marker='D', linestyle='-.', label='improved')

    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(0.056, 0.454), loc=2, borderaxespad=0)
    # plt.ylim(ymin=0, ymax=1.0)
    plt.xlabel('sample point t', fontsize=12)
    plt.ylabel('TPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'ratio2_TPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.savefig(os.path.join(pre, 'ratio2_TPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'ratio2_TPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.clf()

    with open(os.path.join(pre, os.path.join('proposed', 'ratio_original_FPRS.json')), 'r') as FR:
        y1 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('T-online', 'ratio_FPRS.json')), 'r') as FR:
        y2 = json.loads(FR.read())
    with open(os.path.join(pre, os.path.join('proposed', 'ratio_n_FPRS.json')), 'r') as FR:
        y3 = json.loads(FR.read())

    plt.figure(figsize=(10.5, 6.5))
    plt.plot(range(len(x)), y1, color='#3d5d46', marker='o', linestyle='-', label='original')
    plt.plot(range(len(x)), y2, color='#8dabb6', marker='s', linestyle='--', label='T-online')
    plt.plot(range(len(x)), y3, color='#e56a6c', marker='D', linestyle='-.', label='improved')

    plt.xticks(range(len(x)), x, fontsize=12)
    plt.legend(bbox_to_anchor=(0.056, 0.904), loc=2, borderaxespad=0)
    # plt.ylim(ymin=0, ymax=1.0)
    plt.xlabel('sample point t', fontsize=12)
    plt.ylabel('FPR', fontsize=12)
    plt.savefig(os.path.join(pre, 'ratio2_FPRS.png'), bbox_inches='tight', dpi=300, format='png')
    plt.savefig(os.path.join(pre, 'ratio2_FPRS.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join(pre, 'ratio2_FPRS.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.close()


def locations():
    length = 8000
    x = range(length)
    dataset_name = "CERNET"
    pre = os.path.join(os.path.join('results', dataset_name))
    with open(os.path.join(pre, os.path.join('proposed', 'false_locations.json')), 'r') as FR:
        y = json.loads(FR.read())

    st = {}
    for i in range(len(y)):
        for ele in y[i]:
            loc = str(ele[0]) + ' ' + str(ele[1])
            if loc in st:
                st[loc] += 1
            else:
                st[loc] = 1

    st = sorted(st.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
    for item in st:
        print(item[0]+':'+str(item[1]))

    data = np.load('data/CERNET/tensor.npy')
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    for t in range(length):
        y1.append(data[0, 2, t])
        y2.append(data[0, 0, t])
        y3.append(data[0, 11, t])
        y4.append(data[12, 11, t])
        y5.append(data[2, 0, t])
        y6.append(data[12, 0, t])
    # x = []
    # y = []
    # z = []
    # for t in range(2000):
    #     x.append(list(range(15)))
    #     y.append(np.ones(15)*t)
    #     z.append(data[:, :, t].reshape(14*14, )[15:30])

    # fig = plt.figure(figsize=(32, 24))
    # ax = fig.gca(projection='3d')  # get current axes，且坐标轴是3d的
    # ax.scatter(x, y, z, c='red', marker='o', label='points')
    # ax.set_xlabel("X axis")
    # ax.set_ylabel("Y axis")
    # ax.set_zlabel("Z axis")
    plt.figure(figsize=(10.5, 6.5))
    plt.plot(range(len(x)), y1, color='#FF0000', linestyle='-', label='0-2')
    plt.plot(range(len(x)), y2, color='#FF7F00', linestyle='--', label='0-0')
    plt.plot(range(len(x)), y3, color='#FFFF00', linestyle='-.', label='0-11')
    plt.plot(range(len(x)), y4, color='#00FF00', linestyle=':', label='12-11')
    plt.plot(range(len(x)), y5, color='#00FFFF', linestyle=':', label='2-0')
    plt.plot(range(len(x)), y6, color='#0000FF', linestyle=':', label='12-0')
    plt.legend()
    plt.xlabel('sample point t', fontsize=12)
    plt.ylabel('traffic volume', fontsize=12)
    plt.savefig(os.path.join('results', 'locations.pdf'), bbox_inches='tight', dpi=300, format='pdf')
    plt.savefig(os.path.join('results', 'locations.eps'), bbox_inches='tight', dpi=300, format='eps')
    plt.savefig(os.path.join('results', 'locations.png'), bbox_inches='tight',  dpi=300, format='png')



if __name__ == '__main__':
    #plot_mu('Abilene')
    # plot_mu('GEANT')
    # plot_mu('CERNET')
    # plot_sigma('Abilene')
    # plot_sigma('GEANT')
    # plot_sigma('CERNET')
    # #
    # plot_ratio('Abilene')
    # plot_ratio('GEANT')
    # plot_ratio('CERNET')
    # #
    #plot_time('Abilene')
    #plot_time('GEANT')
    #plot_time('CERNET')
    #
    #plot_R('Abilene')
    #plot_R('GEANT')
    #plot_R('CERNET')

    #plot_mu_n()
    #plot_sigma_n()
    #plot_ratio_n()
    #plot_R_n()
    #plot_SNR_n()
    plot_ratio_n2()
    #locations()