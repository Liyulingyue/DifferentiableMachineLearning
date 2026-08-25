"""DifferentiableMachineLearning
A machine learning kit re-implementing classical models as
``torch.nn.Module`` so they can be composed with deep networks.
"""

from . import TimeSeries, ClassicalML, SignalProcessing

__version__ = "0.2.0"
__all__ = ["TimeSeries", "ClassicalML", "SignalProcessing", "__version__"]