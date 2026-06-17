"""The model: a single-layer ConvLSTM that predicts the next frame.

Two pieces, on purpose kept separate:
  - ConvLSTMCell:  the reusable building block (convolution + memory). Standalone
                   so that "add hierarchy later" means literally stacking more of
                   these, not rewriting anything.
  - TemporalPredictor: one cell + a readout that turns the cell's memory into a
                   guess of the *next* frame.

Why ConvLSTM and not plain Rao-Ballard: the LSTM's memory (the cell state `c`)
*is* the "predict the next state" rule the classic Rao-Ballard 1999 model lacks.
The convolution lets it see space (a single pixel can't tell you which way
something moved); the memory lets it carry the past forward. Together that is
exactly "learn the dynamics," which is what makes this more than a motion detector.
"""

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    """One step of a convolutional LSTM (Shi et al., 2015).

    Same equations as a normal LSTM, but every gate is computed by a *convolution*
    instead of a matrix multiply, so the hidden state keeps its 2-D image shape.
    """

    def __init__(self, in_channels, hidden_channels, kernel=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        # One conv produces all four gates at once. It looks at the new input AND
        # the previous hidden state stacked together (that's `in_channels + hidden_channels`).
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels,
                              kernel, padding=kernel // 2) # A normal conv layer: self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)


    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))   # look at input + memory
        input_gate, forget_gate, output_gate, candidate = gates.chunk(4, dim=1)
        input_gate  = input_gate.sigmoid()            # how much new content to write
        forget_gate = forget_gate.sigmoid()           # how much old memory to keep
        output_gate = output_gate.sigmoid()           # how much memory to expose as output
        candidate   = candidate.tanh()                # the new content itself
        c = forget_gate * c + input_gate * candidate  # update long-term memory
        h = output_gate * c.tanh()                    # expose part of it as output
        return h, c


class TemporalPredictor(nn.Module):
    """One ConvLSTM layer + a 1x1 conv that reads the memory out as a frame guess.

    Holds its own (h, c) memory across calls so the caller just feeds frames one by
    one. forward(x) returns the model's prediction of the frame that comes *after* x.
    """

    def __init__(self, in_channels=1, hidden_channels=32, kernel=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel)
        self.readout = nn.Conv2d(hidden_channels, in_channels, 1)  # memory -> predicted pixels - THIS IS ANOTHER CONVOLUTION needed to convert the hidden state values into an actual predicted picture.
        self.hidden_channels = hidden_channels
        self.h = self.c = None                          # memory, created on first frame

    def reset(self):
        """Forget everything learned-so-far about the current scene's motion."""
        self.h = self.c = None

    def forward(self, x):
        if self.h is None:                              # first frame: start memory at zero
            b, _, height, wid = x.shape                    # Throw away channel count...
            self.h = x.new_zeros(b, self.hidden_channels, height, wid)
            self.c = x.new_zeros(b, self.hidden_channels, height, wid)
        self.h, self.c = self.cell(x, self.h, self.c)
        return self.readout(self.h).sigmoid()           # squash to [0,1] pixel range
