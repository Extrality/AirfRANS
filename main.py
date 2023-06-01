import argparse, yaml, os, json, glob
import torch
import train, metrics
from dataset import Dataset
import os.path as osp

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('model', help = 'The model you want to train, choose between MLP, GraphSAGE, PointNet, GUNet', type = str)
parser.add_argument('-n', '--nmodel', help = 'Number of trained models for standard deviation estimation (default: 1)', default = 1, type = int)
parser.add_argument('-w', '--weight', help = 'Weight in front of the surface loss (default: 1)', default = 1, type = float)
parser.add_argument('-t', '--task', help = 'Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)', default = 'full', type = str)
parser.add_argument('-s', '--score', help = 'If you want to compute the score of the models on the associated test set. (default: 0)', default = 0, type = int)
args = parser.parse_args()

with open('Dataset/manifest.json', 'r') as f:
    manifest = json.load(f)

manifest_train = manifest[args.task + '_train']
test_dataset = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
n = int(.1*len(manifest_train))
train_dataset = manifest_train[:-n]
val_dataset = manifest_train[-n:]

# if os.path.exists('Dataset/train_dataset'):
#     train_dataset = torch.load('Dataset/train_dataset')
#     val_dataset = torch.load('Dataset/val_dataset')
#     coef_norm = torch.load('Dataset/normalization')
# else:
train_dataset, coef_norm = Dataset(train_dataset, norm = True, sample = None)
# torch.save(train_dataset, 'Dataset/train_dataset')
# torch.save(coef_norm, 'Dataset/normalization')
val_dataset = Dataset(val_dataset, sample = None, coef_norm = coef_norm)
# torch.save(val_dataset, 'Dataset/val_dataset')

# Cuda
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

with open('params.yaml', 'r') as f: # hyperparameters of the model
    hparams = yaml.safe_load(f)[args.model]

from models.MLP import MLP
models = []
for i in range(args.nmodel):
    encoder = MLP(hparams['encoder'], batch_norm = False)
    decoder = MLP(hparams['decoder'], batch_norm = False)

    if args.model == 'GraphSAGE':
        from models.GraphSAGE import GraphSAGE
        model = GraphSAGE(hparams, encoder, decoder)
    
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        model = PointNet(hparams, encoder, decoder)

    elif args.model == 'MLP':
        from models.NN import NN
        model = NN(hparams, encoder, decoder)

    elif args.model == 'GUNet':
        from models.GUNet import GUNet
        model = GUNet(hparams, encoder, decoder)    

    
    log_path = osp.join('metrics', args.task, args.model) # path where you want to save log and figures    
    model = train.main(device, train_dataset, val_dataset, model, hparams, log_path, 
                criterion = 'MSE_weighted', val_iter = 10, reg = args.weight, name_mod = args.model, val_sample = True)
    models.append(model)
torch.save(models, osp.join('metrics', args.task, args.model, args.model))

if bool(args.score):
    s = args.task + '_test' if args.task != 'scarce' else 'full_test'
    coefs = metrics.Results_test(device, [models], [hparams], coef_norm, path_in = 'Dataset', path_out = 'scores', n_test = 3, criterion = 'MSE', s = s)
    # models can be a stack of the same model (for example MLP) on the task s, if you have another stack of another model (for example GraphSAGE)
    # you can put in model argument [models_MLP, models_GraphSAGE] and it will output the results for both models (mean and std) in an ordered array.
    np.save(osp.join('scores', args.task, 'true_coefs'), coefs[0])
    np.save(osp.join('scores', args.task, 'pred_coefs_mean'), coefs[1])
    np.save(osp.join('scores', args.task, 'pred_coefs_std'), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(osp.join('scores', args.task, 'true_surf_coefs_' + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join('scores', args.task, 'surf_coefs_' + str(n)), file)
    np.save(osp.join('scores', args.task, 'true_bls'), coefs[5])
    np.save(osp.join('scores', args.task, 'bls'), coefs[6])
