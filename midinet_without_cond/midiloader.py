import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import encoder

class MidiDataset(Dataset):
    """A dataset of midifiles"""

    def __init__(self, root_dir, random = False):
        """
        Args:
        root_dir (string): Directory with all the MIDI files
        random (bool): Add a bit of random noise
        """
        self.root_dir = root_dir
        bars = []
        for file in tqdm(os.scandir(root_dir)):
            x = encoder.file_to_dictionary(file.path)['Voice 1']
            bars += x
        bars = np.array(bars, dtype=float)
        if random:
            bars += np.random.randn(bars.shape[0],bars.shape[1],bars.shape[2])/10
        bars[bars >= 1] = 1
        bars[bars <= 0] = 0
        bars = bars.reshape(-1, 1, 48, 128)[:(10848-10848%128)]
        self.data = torch.from_numpy(bars).type(torch.FloatTensor)
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

if __name__=='__main__':
    dataset = MidiDataset("../data")
