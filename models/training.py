import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from tqdm import tqdm


# creation of the model class
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

def data_loading(opt):
    options_dict = {'nn': '../processed_ntuples/transverse_helicity/chunk_nonu_data.npy',
                    'ny': '../processed_ntuples/transverse_helicity/chunk_wnu_data.npy',
                    'yn': '../processed_ntuples/chunking/chunk_nonu_data.npy',
                    'yy': '../processed_ntuples/chunking/chunk_wnu_data.npy'}
    data = np.load(options_dict[opt])
    train, val = train_test_split(data, random_state=137)
    # copy in dedicated arrays
    flags_train = train[:, -1]
    flags_val = val[:, -1]
    # delete from the sets
    train = train[:, :-1]
    val = val[:, :-1]
    # tensorize the data, so that pytorch doesn't whine
    train = torch.tensor(train, dtype=torch.float32)
    val = torch.tensor(val, dtype=torch.float32)
    flags_train = torch.tensor(flags_train, dtype=torch.float32)
    flags_val = torch.tensor(flags_val, dtype=torch.float32)
    return [train, val, flags_train, flags_val]

def main():

    if len(sys.argv) != 2:
        raise ValueError("Incorrect number of arguments. Usage: python training.py nn/ny/yn/yy (the first y/n is for the rpt variable and the second one for neutrino features)")
    
    feats_option = sys.argv[1] # nn, ny, yn, yy; the first y/n is for the rpt variable and the second one for neutrino features
    print('Starting the program')
    
    # data
    train, val, flags_train, flags_val = data_loading(feats_option)
    print('Data loaded and splitted')

    # model prep
    model = myModel(feats_option)
    loss_function = nn.BCELoss() # CrossEntropyLoss but for just one class
    learning_rate = 5.e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=5, delta=0.01)
    print('Model prepared')
    
    # training
    print('Starting the training')
    n_epochs = 60 # just a large upper limit, to let the earlystopping intervene
    batch_size = 100
    
    losses = []
    val_losses = []
    accuracies = []
    val_accuracies = []
    batch_start = torch.arange(0, len(train), batch_size)
    for epoch in range(n_epochs):
        model.train()
        with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                loss_sum = 0.
                acc_sum = 0.
                for start in bar:
                    # take a batch
                    train_batch = train[start:start+batch_size]
                    flags_train_batch = flags_train[start:start+batch_size]
                    # forward pass
                    outputs = model(train_batch)
                    loss = loss_function(outputs, flags_train_batch.unsqueeze(0).T)
                    loss_sum += loss
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    acc = (outputs.round().T == flags_train_batch).float().mean()
                    bar.set_postfix(
                        loss=float(loss),
                        acc=float(acc)
                    )
                    acc_sum += float(acc)
        loss = loss_sum / len(bar)
        losses.append(loss)
        acc = acc_sum / len(bar)
        accuracies.append(acc)
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val)
            val_loss = loss_function(val_outputs, flags_val.unsqueeze(0).T)
            val_losses.append(val_loss)
            val_accuracy = float((val_outputs.round().T == flags_val).float().mean())
            val_accuracies.append(val_accuracy)
        # Early Stopping
        if early_stopping.early_stop(val_loss):
            print(f'Early stopped at epoch: {epoch}')
            n_epochs = epoch+1
            break
        print(f'Epoch: {epoch}, Loss: {loss:.3f}, Validation Loss: {val_loss:.3f}')
    
    print('Training finished')

    # numpyzing the tensor and saving
    label_dict = {'nn': 'norptnonu', 'ny': 'norptwnu', 'yn': 'wrptnonu', 'yy': 'wrptwnu'}
    model.eval()
    with torch.no_grad():
        train_outputs = model(train)
    np.save('results/'+label_dict[feats_option]+'-loss.npy', np.stack([[l.detach().numpy() for l in losses], [l.detach().numpy() for l in val_losses]]))
    np.save('results/'+label_dict[feats_option]+'-acc.npy', np.stack([accuracies, val_accuracies]))
    np.save('results/'+label_dict[feats_option]+'-val_output.npy', val_outputs.detach().numpy())
    np.save('results/'+label_dict[feats_option]+'-train_output.npy', train_outputs.detach().numpy())
    np.save('results/'+label_dict[feats_option]+'-params.npy', [n_epochs, batch_size, learning_rate])
    torch.save(model, 'results/'+label_dict[feats_option]+'-model.pt')


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

if __name__ == "__main__":
	main()