"""Contiguous host storage for opt-in dense binary feature capture."""
import torch


class DenseFeatureStore:
    def __init__(self, spec):
        self.positions = spec.positions
        self.index = {p: i for i,p in enumerate(spec.positions)}
        self.layers = spec.layers
        self.buffers = {}
        self.present = {}
    def __contains__(self, key):
        layer, position = key
        return layer in self.present and position in self.index and bool(self.present[layer][self.index[position]])
    def add(self, layer, positions, tensor):
        if layer not in self.buffers:
            self.buffers[layer] = torch.empty((len(self.positions),tensor.shape[1]),dtype=tensor.dtype,device='cpu')
            self.present[layer] = torch.zeros(len(self.positions),dtype=torch.bool)
        ix = torch.tensor([self.index[p] for p in positions],dtype=torch.long)
        new = ~self.present[layer][ix]
        if bool(new.all()):
            # Each prefill segment is normally a contiguous slice of requested positions.
            if len(ix) and int(ix[-1])-int(ix[0])+1 == len(ix):
                self.buffers[layer][int(ix[0]):int(ix[-1])+1].copy_(tensor)
            else: self.buffers[layer].index_copy_(0,ix,tensor)
        elif bool(new.any()): self.buffers[layer].index_copy_(0,ix[new],tensor[new])
        self.present[layer][ix] = True
    @property
    def complete(self):
        return all(layer in self.present and bool(self.present[layer].all()) for layer in self.layers)
