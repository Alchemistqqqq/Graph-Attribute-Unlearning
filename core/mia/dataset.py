from torch.utils.data import Dataset

class AttackDataset(Dataset):
    def __init__(self, data, labels, transform):

        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.labels)