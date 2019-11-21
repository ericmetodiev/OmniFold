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

# default filenames
FILENAMES = {
    'Herwig': 'Herwig_Preprocessed.pickle',
    'Pythia21': 'Pythia21_Preprocessed.pickle',
    'Pythia25': 'Pythia25_Preprocessed.pickle',
    'Pythia26': 'Pythia26_Preprocessed.pickle',
    'Pythia26-0': 'Pythia26_Preprocessed_0.pickle',
    'Pythia26-1': 'Pythia26_Preprocessed_1.pickle',
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
    parser.add_argument('--dataset-mc', '-mc', choices=FILENAMES.keys(), default='Pythia26')
    parser.add_argument('--dataset-data', '-data', choices=FILENAMES.keys(), default='Herwig')

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
    mc_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_mc]), allow_pickle=True)
    real_preproc = np.load(os.path.join(args.data_path, FILENAMES[args.dataset_data]), allow_pickle=True)
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
        fitargs = {'batch_size': args.batch_size, 'epochs': args.epochs, 'verbose': args.verbose,
                   'weight_clip_min': args.weight_clip_min, 'weight_clip_max': args.weight_clip_max}

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
