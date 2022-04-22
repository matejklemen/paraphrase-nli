from torch.utils.data import Dataset


class TransformersSeqPairDataset(Dataset):
    def __init__(self, **kwargs):
        self.valid_attrs = []
        for attr, values in kwargs.items():
            self.valid_attrs.append(attr)
            setattr(self, attr, values)

        assert len(self.valid_attrs) > 0

    def __getitem__(self, item):
        return {k: getattr(self, k)[item] for k in self.valid_attrs}

    def __len__(self):
        return len(getattr(self, self.valid_attrs[0]))
