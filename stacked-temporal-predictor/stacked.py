"""A stack of ConvLSTM cells that predicts the next frame.

Same online next-frame setup as temporal-predictor, but with several ConvLSTM cells
stacked: each cell's hidden state is the input to the cell above it. Only the top
cell's memory is read out into a predicted frame, and there is a single error signal
at the pixel output -- the lower cells receive no error of their own. Every layer
keeps the full spatial resolution and the same number of channels. This is the
"deeper, not hierarchical" baseline; letting errors flow upward is a later project.
"""

import os
import sys

import torch
from torch import nn

# Reuse the ConvLSTMCell already built for temporal-predictor. Its folder name has a
# hyphen, so it cannot be imported as a package -- put the folder on the path and
# import the module by its (hyphen-free) filename.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "temporal-predictor"))
from convlstm import ConvLSTMCell


class StackedPredictor(nn.Module):
    """num_layers ConvLSTM cells stacked, with one readout on the top cell.

    Holds its own (h, c) memory for every layer across calls, so the caller just
    feeds frames one by one. forward(x) returns the prediction of the frame after x.
    """

    def __init__(self, in_channels=1, hidden_channels=32, num_layers=2, kernel=3):
        super().__init__()
        # The bottom cell reads the frame; every cell above reads the hidden state
        # of the cell below it, so their input channel counts differ only at layer 0.
        layer_in_channels = [in_channels] + [hidden_channels] * (num_layers - 1)
        self.cells = nn.ModuleList(
            ConvLSTMCell(c_in, hidden_channels, kernel) for c_in in layer_in_channels
        )
        self.readout = nn.Conv2d(hidden_channels, in_channels, 1)  # top memory -> predicted pixels
        self.hidden_channels = hidden_channels
        self.h = self.c = None                      # per-layer memory, created on first frame

    def reset(self):
        """Forget everything learned-so-far about the current scene's motion."""
        self.h = self.c = None

    def detach_memory(self):
        """Cut memory loose from the old graph so backprop spans one step (truncated
        BPTT, window 1) and the graph does not grow without bound."""
        if self.h is not None:
            self.h = [h.detach() for h in self.h]
            self.c = [c.detach() for c in self.c]

    def forward(self, x):
        if self.h is None:                          # first frame: start every layer at zero
            b, _, height, wid = x.shape
            self.h = [x.new_zeros(b, self.hidden_channels, height, wid) for _ in self.cells]
            self.c = [x.new_zeros(b, self.hidden_channels, height, wid) for _ in self.cells]

        signal = x
        for i, cell in enumerate(self.cells):
            self.h[i], self.c[i] = cell(signal, self.h[i], self.c[i])
            signal = self.h[i]                      # this layer's output feeds the next one up
        return self.readout(self.h[-1]).sigmoid()   # squash to [0,1] pixel range
