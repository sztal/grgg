from types import SimpleNamespace

options = SimpleNamespace()
options.logspace: bool = True  # use logarithmic relation scores in kernels
options.eps: float = 1e-6  # numerical precision for relation scores calculations
options.sample_batch_size: int = 1000  # batch size for GRGG distance calculations
