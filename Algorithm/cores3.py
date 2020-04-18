from proposed3 import *

Rs = {'Abilene': 2, 'GEANT': 12, 'CERNET': 6}
inits = {'Abilene': 'rand', 'GEANT': 'rand', 'CERNET': 'rand'}


def evaluate(dataset_name, ed, fraction, mu, sigma, R, SNR=None, distribution="Gaussian", init='rand'):
    # Run BayesCP
    Y, outliers_p = generator(dataset_name, fraction, mu, sigma, SNR=SNR, distribution=distribution)
    if ed == None:
        ed = Y.shape[2]
    Y = Y[:, :, 0:ed]
    outliers_p = outliers_p[:, :, 0:ed]
    model = BCPF_IC(Y=Y, outliers_p=outliers_p, maxRank=R, maxiters=30, tol=1e-4, verbose=False, init=init)
    return model


def eval_time(dataset_name, ed):
    # Run BayesCP
    model = evaluate(dataset_name, ed, 0.1, 0, 0.1, Rs[dataset_name])
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/TPRS.json'), 'w') as FD:
        FD.write(json.dumps(model['TPRS']))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/FPRS.json'), 'w') as FD:
        FD.write(json.dumps(model['FPRS']))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/RSE.json'), 'w') as FD:
        FD.write(json.dumps(model['RSE']))


def eval_R(dataset_name):
    TPRS = [[] for _ in range(2, 51, 2)]
    FPRS = [[] for _ in range(2, 51, 2)]
    for i in range(10):
        for R in range(2, 51, 2):
            model = evaluate(dataset_name, 8000, 0.1, 0, 0.1, R)
            TPRS[i].append(model['precision'])
            FPRS[i].append(model['FPR'])

    TPRS_mean = []
    FPRS_mean = []
    for i in range(len(range(2, 51, 2))):
        sum = 0
        sum2 = 0
        for j in range(10):
            sum += TPRS[j][i]
            sum2 += FPRS[j][i]
        TPRS_mean.append(sum / 10)
        FPRS_mean.append(sum2 / 10)

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/R_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS_mean))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/R_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS_mean))


def eval_mu(dataset_name, init="ml"):
    TPRS = []
    FPRS = []
    mus = [0.01, 0.05, 0.1, 0.5, 1]
    mus2 = [0.01, 0.05]
    for mu in mus:
        model = evaluate(dataset_name, 8000, 0.1, mu, 0.1, Rs[dataset_name], init=init)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/mu_n'+init+'_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/mu_n'+init+'_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_sigma(dataset_name, init='ml'):
    TPRS = []
    FPRS = []
    sigmas = [0.01, 0.05, 0.1, 0.5, 1]
    for sigma in sigmas:
        model = evaluate(dataset_name, 8000, 0.1, 0, sigma, Rs[dataset_name], init=init)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/sigma_n'+init+'_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/sigma_n'+init+'_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_ratio(dataset_name, distribution="Gaussian", init='rand'):
    TPRS = []
    FPRS = []

    for fraction in range(1, 6, 1):
        model = evaluate(dataset_name, 8000, fraction / 100, 0, 0.5, Rs[dataset_name], distribution=distribution, init=init)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])
        if fraction == 10:
            with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/false_locations.json'), 'w') as FD:
                FD.write(json.dumps(model['false_locations']))

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/ratio_'+distribution+'_'+init+'_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/ratio_'+distribution+'_'+init+'_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_ratio2(dataset_name, distribution="Gaussian"):
    TPRS = []
    FPRS = []
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5]
    for fraction in fractions:
        model = evaluate(dataset_name, 8000, fraction, 0, 1, Rs[dataset_name], distribution=distribution)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/ratio2_'+distribution+'_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/ratio2_'+distribution+'_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))


def eval_SNR(dataset_name, distribution="Gaussian", init="ml"):
    TPRS = []
    FPRS = []
    SNRs = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
    for SNR in SNRs:
        model = evaluate(dataset_name, 8000, 0.1, 0, 0.1, Rs[dataset_name], SNR, distribution=distribution, init=init)
        TPRS.append(model['precision'])
        FPRS.append(model['FPR'])
        print(SNR)

    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/SNR_'+distribution+'_'+init+'_TPRS.json'), 'w') as FD:
        FD.write(json.dumps(TPRS))
    with open(os.path.join(os.path.join('../../results', dataset_name), 'proposed/SNR_'+distribution+'_'+init+'_FPRS.json'), 'w') as FD:
        FD.write(json.dumps(FPRS))