import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
def bandpass(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq 
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = 'band')
    return lfilter(b, a, data)

def apply_band_filters(data, fs=250, n_bands=None):

    all_bands = [
        ('delta', (1, 4)),
        ('theta', (4, 8)),
        ('alpha', (8, 13)),
        ('beta', (13, 30)),
        ('gamma', (30, 45)),
    ]

    try:
        n_bands = int(n_bands)
    except:
        print(f"[BandFilter] Cannot convert n_bands to int: {n_bands}")
        n_bands = None

    selected_bands = all_bands[:n_bands] if n_bands is not None else all_bands
    print(f"[BandFilter] Bands selected: {[name for name, _ in selected_bands]}")

    filtered = []

    for name, (low, high) in selected_bands:
        band_data = np.array([
            np.array([bandpass(chan, low, high, fs) for chan in trial]) for trial in data
        ])
        filtered.append(band_data)

    return np.stack(filtered, axis=1)

class Source_CustomDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.use_multi_source_align = getattr(args, "use_multi_source_align", False)
        self.load_data()
        self.torch_form()

    def load_data(self):
        self.X = [] 
        self.y = []
        self.domain_labels = []

        for s in self.args.source_subjects:
            X_train = np.load(f"./data/S{s:02}_train_X.npy")
            y_train = np.load(f"./data/S{s:02}_train_y.npy")
            X_test = np.load(f"./data/S{s:02}_test_X.npy")
            y_test = np.load(f"./data/S{s:02}_test_y.npy")

            X = np.concatenate([X_train, X_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)

            if len(X.shape) <= 3:
                X = np.expand_dims(X, axis=1)
            X = np.squeeze(X, axis=1)

            # label filtering 추가 👇
            if self.args.labels != 'all':
                labels = np.array(self.args.labels, dtype = int)
                mask = np.isin(y, labels)
                X = X[mask]
                y = y[mask]

            self.X.append(X)
            self.y.append(y)
            self.domain_labels.append(np.full(len(X), s-1, dtype=int))  # source domain = subject index

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        self.domain_labels = np.concatenate(self.domain_labels, axis=0)

        if self.args.n_bands > 1:
            self.X = apply_band_filters(self.X, n_bands=self.args.n_bands, fs=250)
        else:
            self.X = np.expand_dims(self.X, axis=1)
        print(f"[Check Labels] Unique labels after filtering: {np.unique(self.y)}")
        assert np.all(np.isin(self.y, [0,1,2,3])), f"Found invalid labels: {np.unique(self.y)}"

    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)
        self.domain_labels = torch.LongTensor(self.domain_labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.use_multi_source_align:
            return [self.X[idx], self.y[idx], self.domain_labels[idx]]
        else:
            return [self.X[idx], self.y[idx]]

def source_data_loader(args):
    print("[Load source data]")

    # Load train data
    trainset = Source_CustomDataset(args, mode='train')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # Load val data
    valset = Source_CustomDataset(args, mode='val')
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Print
    print(f"source_train_set size: {train_loader.dataset.X.shape}")
    print(f"source_val_set size: {val_loader.dataset.X.shape}")
    print("")
    return train_loader, val_loader, None


class Target_CustomDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.use_multi_source_align = getattr(args, "use_multi_source_align", False)
        self.load_data()
        self.torch_form()

    def load_data(self):
        self.X = [] 
        self.y = []
        self.domain_labels = [] 

        s = self.args.target_subject

        if self.mode in ['train', 'val']:
            X = np.load(f"./data/S{s:02}_train_X.npy")
            y = np.load(f"./data/S{s:02}_train_y.npy")

            if len(X.shape) <= 3:
                X = np.expand_dims(X, axis=1)
            X = np.squeeze(X, axis=1)

            # label filtering 추가 👇
            if self.args.labels != 'all':
                mask = np.isin(y, self.args.labels)
                X = X[mask]
                y = y[mask]

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

            if self.mode == 'train':
                self.X = X_train
                self.y = y_train
                self.domain_labels = np.full(len(X_train), 8, dtype=int)
            else:
                self.X = X_val
                self.y = y_val 
                self.domain_labels = np.full(len(X_val), 8, dtype=int)
        elif self.mode == 'test':
            self.X = np.load(f"./data/S{s:02}_test_X.npy")
            self.y = np.load(f"./answer/S{s:02}_y_test.npy")

            if len(self.X.shape) <= 3:
                self.X = np.expand_dims(self.X, axis=1)
            self.X = np.squeeze(self.X, axis=1)

            # label filtering 추가 👇
            if self.args.labels != 'all':
                mask = np.isin(self.y, self.args.labels)
                self.X = self.X[mask]
                self.y = self.y[mask]

        if isinstance(self.X, list):
            self.X = np.concatenate(self.X, axis=0)
            self.y = np.concatenate(self.y, axis=0)
            
        if self.args.n_bands > 1:
            self.X = apply_band_filters(self.X, n_bands=self.args.n_bands, fs=250)
        else:
            self.X = np.expand_dims(self.X, axis=1)

    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)
        self.domain_labels = torch.LongTensor(self.domain_labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        if self.use_multi_source_align:
            return [self.X[idx], self.y[idx], self.domain_labels[idx]]
        else:
            return [self.X[idx], self.y[idx]]

def target_data_loader(args):
    print("[Load data]")

    # Load train data
    trainset = Target_CustomDataset(args, mode='train')
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # Load val data (from same session 1)
    valset = Target_CustomDataset(args, mode='val')
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    
    test_set = Target_CustomDataset(args, mode='test')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Print
    print(f"train_set size: {train_loader.dataset.X.shape}")
    print(f"val_set size: {val_loader.dataset.X.shape}")
    print(f"test_set size: {test_loader.dataset.X.shape}")
    print("")
    return train_loader, val_loader, test_loader
