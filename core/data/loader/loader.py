import dgl
import copy
import torch
import numpy as np

from dgl.dataloading.dataloader import CollateWrapper
from dgl.dataloading.dataloader import create_tensorized_dataset      

class DataLoader(torch.utils.data.DataLoader):

    def __init__(self,
                 graph,
                 batch_size,
                 indices,
                 graph_sampler,
                 device=None,
                 shuffle=False,
                 drop_last=False,
                 **kwargs):
        """
        Custom dgl dataloader for handling UnlearnNodeNeighborSampler
        
        """

        self.graph = graph
        self.indices = indices
        self.graph_sampler = graph_sampler
        self.num_workers = kwargs.get("num_workers", 0)
        kwargs["batch_size"] = None

        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())        

        # Device check
        if self.graph.device != indices.device:
            raise ValueError("Graph and indices must be at sme device")

        # Graph check
        if self.graph.device == "cuda" and self.num_workers > 0:
            raise ValueError("GPU dataloading does not support multi-workers")
        self.graph.create_formats_()

        # indices check
        if torch.is_tensor(indices):
            self.dataset = create_tensorized_dataset(
                indices=indices,
                batch_size=batch_size,
                drop_last=drop_last,
                shuffle=shuffle,
                use_ddp = False,
                ddp_seed = 0,
                use_shared_memory = False
            )

        super().__init__(
            self.dataset,
            collate_fn = CollateWrapper(
                self.graph_sampler.sample,
                graph,
                self.device
            ),
            batch_size=None,
            pin_memory=False,
            **kwargs
        )

    
    def __iter__(self):
        if  self.shuffle:
            self.dataset.shuffle()
        
        num_threads =torch.get_num_threads() if self.num_workers > 0 else None
        return _prefetchingIter(
            self, super().__iter__(), num_threads=num_threads
        )


def recursive_apply(data, fn, *args, **kwargs):
    """Recursively apply a function to every element in a container.

    If the input data is a list or any sequence other than a string, returns a list
    whose elements are the same elements applied with the given function.

    If the input data is a dict or any mapping, returns a dict whose keys are the same
    and values are the elements applied with the given function.

    If the input data is a nested container, the result will have the same nested
    structure where each element is transformed recursively.

    The first argument of the function will be passed with the individual elements from
    the input data, followed by the arguments in :attr:`args` and :attr:`kwargs`.

    Parameters
    ----------
    data : any
        Any object.
    fn : callable
        Any function.
    args, kwargs :
        Additional arguments and keyword-arguments passed to the function.

    Examples
    --------
    Applying a ReLU function to a dictionary of tensors:

    >>> h = {k: torch.randn(3) for k in ['A', 'B', 'C']}
    >>> h = recursive_apply(h, torch.nn.functional.relu)
    >>> assert all((v >= 0).all() for v in h.values())
    """
    if isinstance(data, Mapping):
        return {
            k: recursive_apply(v, fn, *args, **kwargs) for k, v in data.items()
        }
    elif isinstance(data, tuple):
        return tuple(recursive_apply(v, fn, *args, **kwargs) for v in data)
    elif is_listlike(data):
        return [recursive_apply(v, fn, *args, **kwargs) for v in data]
    else:
        return fn(data, *args, **kwargs)
    