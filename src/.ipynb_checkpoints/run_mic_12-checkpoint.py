# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import random
import torch
import pandas as pd
import soundfile as sound
import librosa
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
import models
import numpy as np
from traintest import train, validate
from multiprocessing import Pool

# +
sr = 16000
duration = 3

def _load_data(data):
    wav, y_3, y_16 = data
    stereo, fs = sound.read(wav)
    #print(stereo.shape)
    #if len(stereo)==0: #broken audio exception
     #   stereo = np.ones((sr,2))
        #print('--------')
    stereo = stereo / np.abs(stereo).max()
    stereo = librosa.to_mono(stereo.T)
    
    if fs != sr:
        stereo = librosa.resample(stereo, fs, sr)
    #assert stereo.shape[0] > 16000
    if stereo.shape[0] > sr*duration:
        start = (stereo.shape[0] - sr*duration) // 2
        x = stereo[start:start+sr*duration]
    else:
        x = np.pad(stereo, (0, sr*duration-stereo.shape[0]))
    
    
    
    return x, y_3, y_16

def load_data(data_csv):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels_3 = data_df['3_types'].to_list()
    labels_16 = data_df['16_types'].to_list()
    datas = zip(wavpath, labels_3, labels_16)

    with Pool(32) as p:
        return p.map(_load_data, datas)
    
def split(data, limit):
    dict_16 = dict()
    for d in data:
        wav, type_3, type_16 = d
        if type_16 not in dict_16:
            dict_16[type_16] = list()
        dict_16[type_16].append(wav)
    

    new_data = []
    for t in dict_16:
        indexes = list(range(len(dict_16[t])))
        random.shuffle(indexes)
        for i in indexes[:limit]:
            new_data.append((dict_16[t][i],t[0],t))

    return new_data

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


# -

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--model", type=str, default='ast', help="the model used")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands", "microphone"])

parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=32, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# not used in the formal experiments
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
# the stride used in patch spliting, e.g., for patch size 16*16, a stride of 16 means no overlapping, a stride of 10 means overlap of 6.
parser.add_argument("--fstride", type=int, default=10, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=10, help="soft split time stride, overlap=patch_size-stride")
parser.add_argument('--imagenet_pretrain', help='if use ImageNet pretrained audio spectrogram transformer model', type=ast.literal_eval, default='True')
parser.add_argument('--audioset_pretrain', help='if use ImageNet and audioset pretrained audio spectrogram transformer model', type=ast.literal_eval, default='False')

args = parser.parse_args()

# +
classes_3 = ['C','D','M']
classes_12 = ['C1','C2','C3','C4','D1','D2','D3','D4','D5','M1','M2','M3']
genders = ['full', 'female', 'male']

gender = genders[0]

train_ = load_data(args.data_train)
#dev = load_data(args.dev_csv)
test = load_data(args.data_val)

limit = 5
if limit < 100:
    train_ = split(train_, limit)

print ("=== Number of training data: {}".format(len(train_)))
x_train, y_train_3, y_train_12 = list(zip(*train_))
#x_dev, y_dev_3, y_dev_12 = list(zip(*dev))
x_test, y_test_3, y_test_12 = list(zip(*test))
x_train = np.array(x_train)
#x_dev = np.array(x_dev)
x_test = np.array(x_test)

y_train = y_train_12
#y_dev = y_dev_3
y_test = y_test_12
classes = classes_12

cls2label = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)

y_train = [cls2label[y] for y in y_train]
#y_dev = [cls2label[y] for y in y_dev]
y_test = [cls2label[y] for y in y_test]
y_train = to_categorical(y_train, num_classes=num_classes)
#y_dev = keras.utils.to_categorical(y_dev, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

#print('train shape')
#print(x_train.shape)
#print(y_train.shape)
#print(x_train[1])
#print(y_train[1])
# -

# transformer based model
if args.model == 'ast':
    print('now train a audio spectrogram transformer model')
    # dataset spectrogram mean and std, used to normalize the input
    norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526], 'microphone': [0,0]}
    target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128, 'microphone':1024}
    # if add noise for data augmentation, only use for speech commands
    noise = {'audioset': False, 'esc50': False, 'speechcommands':True, 'microphone':False}

    audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1],
                  'noise':noise[args.dataset], 'skip_norm': True}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise':False, 'skip_norm': True}

    
    print('data loading ...')
    train_loader = torch.utils.data.DataLoader(
        dataloader.MicDataset(x_train, y_train, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataloader.MicDataset(x_test, y_test, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=target_length[args.dataset], imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size='base384')

print("\nCreating experiment directory: %s" % args.exp_dir)
os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

print('Now starting training for {:d} epochs'.format(args.n_epochs))
#for i, (audio_input, labels) in enumerate(train_loader):
#    print(i, labels)
train(audio_model, train_loader, val_loader, args)

# for speechcommands dataset, evaluate the best model on validation set on the test set
if args.dataset == 'speechcommands':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    # best model on the validation set
    stats, _ = validate(audio_model, val_loader, args, 'valid_set')
    # note it is NOT mean of class-wise accuracy
    val_acc = stats[0]['acc']
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the validation set---------------')
    print("Accuracy: {:.6f}".format(val_acc))
    print("AUC: {:.6f}".format(val_mAUC))

    # test the model on the evaluation set
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stats, _ = validate(audio_model, eval_loader, args, 'eval_set')
    eval_acc = stats[0]['acc']
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print('---------------evaluate on the test set---------------')
    print("Accuracy: {:.6f}".format(eval_acc))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/eval_result.csv', [val_acc, val_mAUC, eval_acc, eval_mAUC])

