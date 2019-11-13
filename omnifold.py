import argparse
import gc
import os
import sys
import time

import energyflow as ef
import numpy as np

# default paths
MACHINES = {
    'voltan': {
        'data_path': '/data1/pkomiske/OmniFold',
        'results_path': '/data1/pkomiske/OmniFold/results'
    },
    'squirrel': {
        'data_path': '/data0/users/pkomiske/OmniFold',
        'results_path': '/data0/users/pkomiske/OmniFold/results'
    },
    'ctp': {
        'data_path': '/Volumes/ganymede/OmniFold',
        'results_path': '/Volumes/ganymede/OmniFold/results'
    }
}

def main(arg_list):

    # parse options, allow global access
    global args
    args = construct_parser(arg_list)

    # this must come before importing tensorflow to get the right GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import energyflow.archs

    # handle names
    if args.unfolding == 'omnifold':
        name = args.name + 'OmniFold_{}_Rep-{}'
    elif args.unfolding == 'manyfold':
        name = args.name + 'ManyFold_DNN_Rep-{}'
    elif args.unfolding == 'unifold':
        name = args.name + 'UniFold_DNN_{}'

    # iteration loop
    for i in range(args.start_iter, args.max_iter):
        if args.unfolding == 'omnifold':
            args.name = name.format(args.omnifold_arch, i)
            train_omnifold(i)
        elif args.unfolding == 'manyfold':
            args.name = name.format(i)
            train_manyfold(i)
        elif args.unfolding == 'unifold':
            args.name = name + '_Rep-{}'.format(i)
            train_unifold(i)

def construct_parser(args):

    parser = argparse.ArgumentParser(description='OmniFold unfolding.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data selection
    parser.add_argument('--machine', '-m', choices=MACHINES.keys(), required=True)
    parser.add_argument('--dataset-mc', '-mc', choices=['Herwig', 'Pythia21', 'Pythia25', 'Pythia26'], default='Pythia26')
    parser.add_argument('--dataset-data', '-data', choices=['Herwig', 'Pythia21', 'Pythia25', 'Pythia26'], default='Herwig')

    # unfolding options
    parser.add_argument('--unfolding', '-u', choices=['omnifold', 'manyfold', 'unifold'], required=True)
    parser.add_argument('--step2-ind', type=int, choices=[0, -2], default=0)
    parser.add_argument('--unfolding-iterations', '-ui', type=int, default=8)
    parser.add_argument('--weight-clip-min', type=float, default=0.)
    parser.add_argument('--weight-clip-max', type=float, default=np.inf)

    # neural network settings
    parser.add_argument('--Phi-sizes', '-sPhi', type=int, nargs='*', default=[100, 100, 256])
    parser.add_argument('--F-sizes', '-sF', type=int, nargs='*', default=[100, 100, 100])
    parser.add_argument('--omnifold-arch', '-a', choices=['PFN'], default='PFN')
    parser.add_argument('--batch-size', '-bs', type=int, default=500)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--gpu', '-g', default='0')
    parser.add_argument('--input-dim', type=int, default=4)
    parser.add_argument('--patience', '-p', type=int, default=10)
    parser.add_argument('--save-best-only', action='store_true')
    parser.add_argument('--save-full-model', action='store_true')
    parser.add_argument('--val-frac', '-val', type=float, default=0.2)
    parser.add_argument('--verbose', '-v', type=int, choices=[0, 1, 2], default=2)

    # training settings
    parser.add_argument('--max-iter', '-i', type=int, default=1)
    parser.add_argument('--name', '-n', default='')
    parser.add_argument('--start-iter', '-si', type=int, default=0)

    p_args = parser.parse_args(args=args)
    p_args.data_path = MACHINES[p_args.machine]['data_path']
    p_args.results_path = MACHINES[p_args.machine]['results_path']

    return p_args

def train_omnifold(i):

    start = time.time()

    # load datasets
    mc_preproc = np.load(os.path.join(args.data_path, args.dataset_mc + '_Preprocessed.pickle'), allow_pickle=True)
    real_preproc = np.load(os.path.join(args.data_path, args.dataset_data + '_Preprocessed.pickle'), allow_pickle=True)
    gen, sim, data = mc_preproc['gen'], mc_preproc['sim'], real_preproc['sim']
    del mc_preproc, real_preproc['sim']

    # pad datasets
    start = time.time()
    sim_data_max_length = max(get_max_length(sim), get_max_length(data))
    gen, sim = pad_events(gen), pad_events(sim, max_length=sim_data_max_length)
    data = pad_events(data, max_length=sim_data_max_length)
    print('Done padding in {:.3f}s'.format(time.time() - start))

    # detector/sim setup
    global X_det, Y_det
    X_det = (np.concatenate((data, sim), axis=0))
    Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(data)), np.zeros(len(sim)))))
    del data, sim

    # gen setup
    global X_gen, Y_gen
    X_gen = (np.concatenate((gen, gen)))
    Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(gen)), np.zeros(len(gen)))))
    del gen

    # specify the model and the training parameters
    model1_fp = os.path.join(args.results_path, 'models',  args.name + '_Iter-{}-Step1')
    model2_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step2')
    Model = getattr(ef.archs, args.omnifold_arch)
    det_args = {'input_dim': args.input_dim, 'Phi_sizes': args.Phi_sizes, 'F_sizes': args.F_sizes, 
                'patience': args.patience, 'filepath': model1_fp, 'save_weights_only': args.save_full_model, 
                'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    mc_args = {'input_dim': args.input_dim, 'Phi_sizes': args.Phi_sizes, 'F_sizes': args.F_sizes, 
               'patience': args.patience, 'filepath': model2_fp, 'save_weights_only': args.save_full_model, 
               'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    fitargs = {'batch_size': args.batch_size, 'epochs': args.epochs, 'verbose': args.verbose}

    # apply the omnifold technique to this one dimensional space
    ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
    wdata = np.ones(ndata)
    winit = ndata/nsim*np.ones(nsim)
    ws = omnifold('X_gen', 'Y_gen', 'X_det', 'Y_det', wdata, winit, (Model, det_args), (Model, mc_args), fitargs, 
                  val=args.val_frac, it=args.unfolding_iterations, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name),
                  delete_global_arrays=True)

    print('Finished OmniFold {} in {:.3f}s'.format(i, time.time() - start))

def load_obs():

    # load datasets
    datasets = {args.dataset_mc: {}, args.dataset_data: {}}
    for dataset,v in datasets.items():
        filepath = '{}/{}_ZJet'.format(args.data_path, dataset)
        
        # load particles
        v.update(np.load(filepath + '.pickle', allow_pickle=True))
        
        # load npzs
        f = np.load(filepath + '.npz')
        v.update({k: f[k] for k in f.files})
        f.close()
        
        # load obs
        f = np.load(filepath + '_Obs.npz')
        v.update({k: f[k] for k in f.files})
        f.close()

    # choose what is MC and Data in this context
    mc, real = datasets[args.dataset_mc], datasets[args.dataset_data]

    # a dictionary to hold information about the observables
    obs = {
        'Mass': {'func': lambda dset, ptype: dset[ptype + '_jets'][:,3]},
        'Mult': {'func': lambda dset, ptype: dset[ptype + '_mults']},
        'Width': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,1]},
        'Tau21': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,4]/(dset[ptype + '_nsubs'][:,1] + 10**-50)},
        'zg': {'func': lambda dset, ptype: dset[ptype + '_zgs'][:,0]},
        'SDMass': {'func': lambda dset, ptype: np.log(dset[ptype + '_sdms'][:,0]**2/dset[ptype + '_jets'][:,0]**2 + 10**-100)},
        'LHA': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,0]},
        'e2': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,2]},
        'Tau32': {'func': lambda dset, ptype: dset[ptype + '_nsubs'][:,7]/(dset[ptype + '_nsubs'][:,4] + 10**-50)},
        'Rapidity': {'func': lambda dset, ptype: dset[ptype + '_jets'][:,1]}
    }

    # calculate quantities to be stored in obs
    for obkey,ob in obs.items():
        
        # calculate observable for GEN, SIM, DATA, and TRUE
        ob['genobs'], ob['simobs'] = ob['func'](mc, 'gen'), ob['func'](mc, 'sim')
        ob['truthobs'], ob['dataobs'] = ob['func'](real, 'gen'), ob['func'](real, 'sim')
        print('Done computing', obkey)

    print()
    del mc, real, datasets
    gc.collect()

    return obs

def train_manyfold(i):

    obs = load_obs()

    # which observables to include in manyfold
    obkeys = ['Mass', 'Mult', 'Width', 'Tau21', 'zg', 'SDMass', 'Rapidity']

    start = time.time()
    print('ManyFolding')
    
    # detector/sim setup
    X_det = np.asarray([np.concatenate((obs[obkey]['dataobs'], obs[obkey]['simobs'])) for obkey in obkeys]).T
    Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(obs['Mass']['dataobs'])), np.zeros(len(obs['Mass']['simobs'])))))

    # gen setup
    X_gen = np.asarray([np.concatenate((obs[obkey]['genobs'], obs[obkey]['genobs'])) for obkey in obkeys]).T
    Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(obs['Mass']['genobs'])), np.zeros(len(obs['Mass']['genobs'])))))
    
    # standardize the inputs
    X_det = (X_det - np.mean(X_det, axis=0))/np.std(X_det, axis=0)
    X_gen = (X_gen - np.mean(X_gen, axis=0))/np.std(X_gen, axis=0)

    # specify the model and the training parameters
    model1_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step1')
    model2_fp = os.path.join(args.results_path, 'models', args.name + '_Iter-{}-Step2')
    Model = ef.archs.DNN
    det_args = {'input_dim': len(obkeys), 'dense_sizes': args.F_sizes, 
                'patience': args.patience, 'filepath': model1_fp, 'save_weights_only': args.save_full_model, 
                'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    mc_args = {'input_dim': len(obkeys), 'dense_sizes': args.F_sizes, 
               'patience': args.patience, 'filepath': model2_fp, 'save_weights_only': args.save_full_model, 
               'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
    fitargs = {'batch_size': args.batch_size, 'epochs': args.epochs, 'verbose': args.verbose}

    # apply the unifold technique to this one dimensional space
    ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
    wdata = np.ones(ndata)
    winit = ndata/nsim*np.ones(nsim)
    ws = omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (Model, det_args), (Model, mc_args), 
                  fitargs, val=args.val_frac, it=args.unfolding_iterations, trw_ind=args.step2_ind,
                  weights_filename=os.path.join(args.results_path, 'weights', args.name))

    print('Finished ManyFold {} in {:.3f}s\n'.format(i, time.time() - start))

def train_unifold(i):

    obs = load_obs()

    # UniFold
    for obkey in ['Mass', 'Mult', 'Width', 'Tau21', 'zg', 'SDMass', 'LHA', 'e2', 'Tau32']:
        start = time.time()

        print('Un[i]Folding', obkey)
        ob = obs[obkey]
        ob_filename = args.name.format(obkey)

        # detector/sim setup
        X_det = (np.concatenate((ob['dataobs'], ob['simobs']), axis=0))
        Y_det = ef.utils.to_categorical(np.concatenate((np.ones(len(ob['dataobs'])), np.zeros(len(ob['simobs'])))))

        # gen setup
        X_gen = (np.concatenate((ob['genobs'], ob['genobs'])))
        Y_gen = ef.utils.to_categorical(np.concatenate((np.ones(len(ob['genobs'])), np.zeros(len(ob['genobs'])))))
        
        # standardize the inputs
        X_det = (X_det - np.mean(X_det))/np.std(X_det)
        X_gen = (X_gen - np.mean(X_gen))/np.std(X_gen)

        # specify the model and the training parameters
        model1_fp = os.path.join(args.results_path, 'models',  ob_filename + '_Iter-{}-Step1')
        model2_fp = os.path.join(args.results_path, 'models', ob_filename + '_Iter-{}-Step2')
        Model = ef.archs.DNN
        det_args = {'input_dim': 1, 'dense_sizes': args.F_sizes, 
                    'patience': args.patience, 'filepath': model1_fp, 'save_weights_only': args.save_full_model, 
                    'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
        mc_args = {'input_dim': 1, 'dense_sizes': args.F_sizes, 
                   'patience': args.patience, 'filepath': model2_fp, 'save_weights_only': args.save_full_model, 
                   'modelcheck_opts': {'save_best_only': args.save_best_only, 'verbose': 0}}
        fitargs = {'batch_size': args.batch_size, 'epochs': args.epochs, 'verbose': args.verbose}

        # apply the unifold technique to this one dimensional space
        ndata, nsim = np.count_nonzero(Y_det[:,1]), np.count_nonzero(Y_det[:,0])
        wdata = np.ones(ndata)
        winit = ndata/nsim*np.ones(nsim)
        ws = omnifold(X_gen, Y_gen, X_det, Y_det, wdata, winit, (Model, det_args), (Model, mc_args), 
                      fitargs, val=args.val_frac, it=args.unfolding_iterations, trw_ind=args.step2_ind,
                      weights_filename=os.path.join(args.results_path, 'weights', ob_filename))

        print('Finished UniFold {} for {} in {:.3f}s\n'.format(i, obkey, time.time() - start))

def pad_events(events, val=0, max_length=None):
    event_lengths = [event.shape[0] for event in events]
    if max_length is None:
        max_length = max(event_lengths)
    return np.asarray([np.vstack((event, val*np.ones((max_length - ev_len, event.shape[1]))))
                       for event,ev_len in zip(events, event_lengths)])

def get_max_length(events):
    return max([event.shape[0] for event in events])

# DCTR, reweights positive distribution to negative distribution
# X: features
# Y: categorical labels
# model: model with fit/predict
# fitargs: model fit arguments
def reweight(X, Y, w, model, filepath, fitargs, val_data=None):

    # permute the data, fit the model, and get preditions
    #perm = np.random.permutation(len(X))
    #model.fit(X[perm], Y[perm], sample_weight=w[perm], **fitargs)
    val_dict = {'validation_data': val_data} if val_data is not None else {}
    model.fit(X, Y, sample_weight=w, **fitargs, **val_dict)
    model.save_weights(filepath)
    preds = model.predict(X, batch_size=10*fitargs.get('batch_size', 500))[:,1]
    
    # concatenate validation predictions into training predictions
    if val_data is not None:
        preds_val = model.predict(val_data[0], batch_size=10*fitargs.get('batch_size', 500))[:,1]
        preds = np.concatenate((preds, preds_val))
        w = np.concatenate((w, val_data[2]))

    w *= np.clip(preds/(1 - preds + 10**-50), args.weight_clip_min, args.weight_clip_max)
    return w

# OmniFold
# X_gen/Y_gen: particle level features/labels
# X_det/Y_det: detector level features/labels, note these should be ordered as (data, sim)
# wdata/winit: initial weights of the data/simulation
# model: model with fit/predict
# fitargs: model fit arguments
# it: number of iterations
# trw_ind: which previous weights to use in second step, 0 means use initial, -2 means use previous
def omnifold(X_gen_i, Y_gen_i, X_det_i, Y_det_i, wdata, winit, det_model, mc_model, fitargs, 
             val=0.2, it=10, weights_filename=None, trw_ind=0, delete_global_arrays=False):

    # get arrays (possibly globally)
    X_gen_arr = globals()[X_gen_i] if isinstance(X_gen_i, str) else X_gen_i
    Y_gen_arr = globals()[Y_gen_i] if isinstance(Y_gen_i, str) else Y_gen_i
    X_det_arr = globals()[X_det_i] if isinstance(X_det_i, str) else X_det_i
    Y_det_arr = globals()[Y_det_i] if isinstance(Y_det_i, str) else Y_det_i
    
    # initialize the truth weights to the prior
    ws = [winit]
    
    # get permutation for det
    perm_det = np.random.permutation(len(winit) + len(wdata))
    invperm_det = np.argsort(perm_det)
    nval_det = int(val*len(perm_det))
    X_det_train, X_det_val = X_det_arr[perm_det[:-nval_det]], X_det_arr[perm_det[-nval_det:]]
    Y_det_train, Y_det_val = Y_det_arr[perm_det[:-nval_det]], Y_det_arr[perm_det[-nval_det:]]

    # remove X_det, Y_det
    if delete_global_arrays:
        del X_det_arr, Y_det_arr
        if isinstance(X_det_i, str):
            del globals()[X_det_i]
        if isinstance(Y_det_i, str):
            del globals()[Y_det_i]
    
    # get an initial permutation for gen and duplicate (offset) it
    nval = int(val*len(winit))
    baseperm0 = np.random.permutation(len(winit))
    baseperm1 = baseperm0 + len(winit)
    
    # training examples are at beginning, val at end
    # concatenate into single train and val perms (shuffle each)
    trainperm = np.concatenate((baseperm0[:-nval], baseperm1[:-nval]))
    valperm = np.concatenate((baseperm0[-nval:], baseperm1[-nval:]))
    np.random.shuffle(trainperm)
    np.random.shuffle(valperm)
    
    # get final permutation for gen (ensured that the same events end up in val)
    perm_gen = np.concatenate((trainperm, valperm))
    invperm_gen = np.argsort(perm_gen)
    nval_gen = int(val*len(perm_gen))
    X_gen_train, X_gen_val = X_gen_arr[perm_gen[:-nval_gen]], X_gen_arr[perm_gen[-nval_gen:]]
    Y_gen_train, Y_gen_val = Y_gen_arr[perm_gen[:-nval_gen]], Y_gen_arr[perm_gen[-nval_gen:]]

    # remove X_gen, Y_gen
    if delete_global_arrays:
        del X_gen_arr, Y_gen_arr
        if isinstance(X_gen_i, str):
            del globals()[X_gen_i]
        if isinstance(Y_gen_i, str):
            del globals()[Y_gen_i]

    # store model filepaths
    model_det_fp, model_mc_fp = det_model[1].get('filepath', None), mc_model[1].get('filepath', None)
    
    # iterate the procedure
    for i in range(it):

        # det filepaths properly
        if model_det_fp is not None:
            model_det_fp_i = model_det_fp.format(i)
            det_model[1]['filepath'] = model_det_fp_i + '_Epoch-{epoch}'
        if model_mc_fp is not None:
            model_mc_fp_i = model_mc_fp.format(i)
            mc_model[1]['filepath'] = model_mc_fp_i + '_Epoch-{epoch}'

        # define models
        model_det = det_model[0](**det_model[1])
        model_mc = mc_model[0](**mc_model[1])

        # load weights if not model 0
        if i > 0:
            model_det.load_weights(model_det_fp.format(i-1))
            model_mc.load_weights(model_mc_fp.format(i-1))
        
        # step 1: reweight sim to look like data
        w = np.concatenate((wdata, ws[-1]))
        w_train, w_val = w[perm_det[:-nval_det]], w[perm_det[-nval_det:]]
        rw = reweight(X_det_train, Y_det_train, w_train, model_det, model_det_fp_i,
                      fitargs, val_data=(X_det_val, Y_det_val, w_val))[invperm_det]
        ws.append(rw[len(wdata):])

        # step 2: reweight the prior to the learned weighting
        w = np.concatenate((ws[-1], ws[trw_ind]))
        w_train, w_val = w[perm_gen[:-nval_gen]], w[perm_gen[-nval_gen:]]
        rw = reweight(X_gen_train, Y_gen_train, w_train, model_mc, model_mc_fp_i,
                      fitargs, val_data=(X_gen_val, Y_gen_val, w_val))[invperm_gen]
        ws.append(rw[len(ws[-1]):])
        
        # save the weights if specified
        if weights_filename is not None:
            np.save(weights_filename, ws)
        
    return ws

if __name__ == '__main__':
    main(sys.argv[1:])