import sys

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from scipy.stats import ks_2samp


# creation of the model class
# (don't know if the sigmoid is defined in the best way possibile; this one just works)
class myModel(nn.Module):
    def __init__(self, opt):
        options_dict = {'nn': 31, 'ny': 37, 'yn': 32, 'yy': 38}
        super(myModel, self).__init__()
        self.dense1 = nn.Linear(options_dict[opt], 64)
        self.BatchNorm1d1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.dense2 = nn.Linear(64, 48)
        self.BatchNorm1d2 = nn.BatchNorm1d(48)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.dense3 = nn.Linear(48, 32)
        self.BatchNorm1d3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.dense4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.dense1(x)
        x = self.BatchNorm1d1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.BatchNorm1d2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.BatchNorm1d3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.dense4(x)
        x = self.sigmoid(x)
        return x

class EarlyStopping:
    def __init__(self, patience=5, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None

    def early_stop(self, validation_loss):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False

def flags_loading(opt):
    options_dict = {'nn': '../processed_ntuples/chunking/chunk_nonu_data.npy',
                    'ny': '../processed_ntuples/chunking/chunk_wnu_data.npy',
                    'yn': '../processed_ntuples/chunking/chunk_nonu_data.npy',
                    'yy': '../processed_ntuples/chunking/chunk_wnu_data.npy'}
    data = np.load(options_dict[opt])
    if opt == 'nn' or opt == 'ny': data = np.concatenate((data[:,0:-2], data[:,-1].reshape(-1,1)), axis=1)
    train, val = train_test_split(data, random_state=137)
    # copy in dedicated arrays
    flags_train = train[:, -1]
    flags_val = val[:, -1]
    return [flags_train, flags_val]

def plotting(feats_option, n_epochs, batch_size, learning_rate, accuracies, val_accuracies, losses, val_losses, train_outputs, val_outputs, flags_train, flags_val):
    label_dict = {'nn': 'norptnonu', 'ny': 'norptwnu', 'yn': 'wrptnonu', 'yy': 'wrptwnu'}
    # accuracies
    plt.figure()
    plt.plot(list(range(n_epochs)), accuracies, label='train')
    plt.plot(list(range(n_epochs)), val_accuracies, label='val')
    plt.title(f'Accuracy\nTraining with {n_epochs} epochs, batch size {batch_size} and learning rate = {learning_rate:.0e}')
    plt.grid(alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(fontsize=12)
    plt.savefig(f'figures/'+label_dict[feats_option]+'-acc.png', dpi=500, bbox_inches='tight')
    plt.close()
    print('Accuracy plot saved')
    # losses
    plt.figure()
    plt.plot(list(range(n_epochs)), losses, label='train')
    plt.plot(list(range(n_epochs)), val_losses, label='val')
    plt.title(f'Loss\nTraining with {n_epochs} epochs, batch size {batch_size} and learning rate = {learning_rate:.0e}')
    plt.grid(alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(fontsize=12)
    plt.savefig(f'figures/'+label_dict[feats_option]+'-loss.png', dpi=500, bbox_inches='tight')
    plt.close()
    print('Loss plot saved')
    # roc
    plt.figure()
    fpr, tpr, _ = roc_curve(flags_val, val_outputs)
    auc = roc_auc_score(flags_val, val_outputs)
    plt.plot(fpr, tpr, label=f'AUC {auc:.3f}')
    plt.title(f'ROC curve\nTraining with {n_epochs} epochs, batch size {batch_size} and learning rate = {learning_rate:.0e}')
    plt.grid(alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(fontsize=14)
    plt.axis('square')
    plt.savefig(f'figures/'+label_dict[feats_option]+'-ROC.png', dpi=500, bbox_inches='tight')
    plt.close()
    print('ROC plot saved')
    # output val
    plt.figure()
    sig_val = val_outputs[flags_val.T.astype(bool)]
    bkg_val = val_outputs[~flags_val.T.astype(bool)]
    plt.hist(sig_val, bins=50, density=True, alpha=0.7, label='sig')
    plt.hist(bkg_val, bins=50, density=True, alpha=0.7, label='bkg')
    plt.title(f'Output of the model, validation data\nTraining with {n_epochs} epochs, batch size {batch_size} and learning rate = {learning_rate:.0e}')
    plt.xlabel('Output')
    plt.ylabel('Normalized counts')
    plt.legend(fontsize=12)
    plt.savefig(f'figures/'+label_dict[feats_option]+'-output_val.png', dpi=500, bbox_inches='tight')
    plt.close()
    print('Output plot saved')
    # output val+train
    plt.figure()
    sig_tr = train_outputs[flags_train.T.astype(bool)]
    bkg_tr = train_outputs[~flags_train.T.astype(bool)]
    plt.hist(sig_tr, bins=50, density=True,  alpha=0.7, label='sig_tr')
    plt.hist(bkg_tr, bins=50, density=True, alpha=0.7, label='bkg_tr')
    plt.hist(sig_val, bins=50, density=True, histtype='step', color='k', label='sig_val')
    plt.hist(bkg_val, bins=50, density=True, histtype='step', color='red', label='bkg_val')
    plt.title(f'Output of the model, both for validation and for training data\nKS p-values: signal {ks_2samp(sig_tr.reshape(1,-1).squeeze(), sig_val.reshape(1,-1).squeeze())[1]:.2f}, '
              +f'background {ks_2samp(bkg_tr.reshape(1,-1).squeeze(), bkg_val.reshape(1,-1).squeeze())[1]:.2f}\nTraining with {n_epochs} epochs, batch size {batch_size} and learning rate = {learning_rate:.0e}')
    plt.yscale('log')
    plt.xlabel('Output')
    plt.ylabel('Normalized counts (log-scale)')
    plt.legend(fontsize=12)
    plt.savefig(f'figures/'+label_dict[feats_option]+'-output_trainval.png', dpi=500, bbox_inches='tight')
    plt.close()
    print('Output full plot saved')



def main():

    if len(sys.argv) != 2:
        raise ValueError("Incorrect number of arguments. Usage: python training.py nn/ny/yn/yy (the first y/n is for the rpt variable and the second one for neutrino features)")

    feats_option = sys.argv[1] # nn, ny, yn, yy; the first y/n is for the rpt variable and the second one for neutrino features
    print('Starting the program')
    
    # starting data loading
    flags_train, flags_val = flags_loading(feats_option)
    
    # resulting data loading
    label_dict = {'nn': 'norptnonu', 'ny': 'norptwnu', 'yn': 'wrptnonu', 'yy': 'wrptwnu'}
    n_epochs, batch_size, learning_rate = np.load('results/'+label_dict[feats_option]+'-params.npy')
    n_epochs = int(n_epochs)
    batch_size = int(batch_size)
    accuracies, val_accuracies= np.load('results/'+label_dict[feats_option]+'-acc.npy')
    losses, val_losses = np.load('results/'+label_dict[feats_option]+'-loss.npy')
    train_outputs = np.load('results/'+label_dict[feats_option]+'-train_output.npy')
    val_outputs = np.load('results/'+label_dict[feats_option]+'-val_output.npy')
    print('Data loaded and splitted')

    # plotting
    plotting(feats_option, n_epochs, batch_size, learning_rate, accuracies, val_accuracies, losses, val_losses, train_outputs, val_outputs, flags_train, flags_val)




# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

if __name__ == "__main__":
	main()