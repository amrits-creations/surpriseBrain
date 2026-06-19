"""Oddball experiment: turn the "it fades" vibe into a measured curve.

Same model, same predict -> error -> step -> re-predict loop as main.py. Only the
input and output change: instead of the camera we feed a predictable synthetic
stimulus, and instead of just watching we LOG how surprised the model is on every
frame.

The standard stimulus comes in two flavours (set STANDARD below):
  - "static": a shape held still. The model only has to predict "next = current",
    which it learns perfectly, so the surprise fades to TRUE BLACK.
  - "moving": a bar sweeping across the frame. The model has to predict where a
    sharp edge will be next -- it can never get that pixel-perfect, so a faint rim
    of error survives and the surprise fades only to DARK GREY, not black. That is
    a real property of a one-step-ahead predictor, not a bug.

Either way: once it has settled, we OMIT the stimulus for a range of frames (the
pattern-breaking deviant). The surprise flares when the expected stimulus vanishes
and again when it returns -- that is the mismatch response.

The live window is the same "input | surprise" view as main.py. When the run
ends, we plot surprise-per-frame: the falling envelope is repetition suppression,
the shaded band is the oddball.

Press 'q' to stop early.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from convlstm import TemporalPredictor

SIZE = 64          # frame is SIZE x SIZE grayscale, like main.py
LR = 5e-3          # learning rate = how fast surprise fades (same knob as main.py)
GAIN = 4.0         # display brightness for the error map (fixed, so learned scenes go dark)

STANDARD = "moving"    # "static" -> fades to true black; "moving" -> fades to dark grey
BAR = 8            # width of the bar, in pixels
STEP = 1           # how far the bar moves each frame (only used when STANDARD == "moving")
STATIC_X = 28      # the bar's fixed column (only used when STANDARD == "static")
TOTAL = 1000       # how many frames the experiment runs for
ODDBALL_LRANGE = 800   # first frame of the deviant (the bar is omitted from here...)
ODDBALL_RRANGE = 820   # ...to here, inclusive


def to_tensor(gray):
    """HxW uint8 image -> 1x1xHxW float tensor in [0, 1]."""
    return torch.from_numpy(gray).float().div(255.0)[None, None]


def stimulus(t):
    """The frame shown at step t.

    Normally a white bar (held still, or swept across and wrapped around). During
    the oddball range we return black instead -- the bar is omitted, which is the
    one thing the model did not predict.
    """
    frame = np.zeros((SIZE, SIZE), np.uint8)
    if ODDBALL_LRANGE <= t <= ODDBALL_RRANGE:
        return frame                                  # deviant: expected bar is missing
    x = STATIC_X if STANDARD == "static" else (t * STEP) % SIZE
    for k in range(BAR):
        frame[:, (x + k) % SIZE] = 255                # draw the bar, wrapping the edge cleanly
    return frame


def main():
    model = TemporalPredictor(in_channels=1, hidden_channels=32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = []        # mean surprise per frame -- the measurement we came for
    pred = None         # our guess for the CURRENT frame, made on the previous iteration
    for t in range(TOTAL):
        gray = stimulus(t)
        x = to_tensor(gray)

        if pred is not None:
            error = x - pred                           # how wrong last step's guess was
            loss = error.pow(2).mean()                 # train on that mistake...
            opt.zero_grad()
            loss.backward()
            opt.step()                                 # ...one online step toward being right
            surprise = error.detach().abs()[0, 0].numpy()
        else:
            surprise = np.zeros((SIZE, SIZE), np.float32)
        history.append(float(surprise.mean()))

        # Truncated BPTT, window 1: cut memory loose so backprop spans one step only.
        if model.h is not None and model.c is not None:
            model.h, model.c = model.h.detach(), model.c.detach()

        pred = model(x)                                # guess the NEXT frame from this one

        # Paint: stimulus beside its surprise map, same layout as main.py.
        smap = np.clip(surprise * GAIN * 255, 0, 255).astype(np.uint8)
        view = np.hstack([cv2.resize(gray, (480, 480), interpolation=cv2.INTER_NEAREST),
                          cv2.resize(smap, (480, 480), interpolation=cv2.INTER_NEAREST)])
        cv2.imshow("stimulus  |  surprise", view)
        if cv2.waitKey(40) & 0xFF == ord('q'):         # ~25 fps so the motion is watchable
            break

    cv2.destroyAllWindows()

    # The deliverable: surprise vs. frame. The falling envelope is repetition
    # suppression; the shaded band is the oddball (flare on omit, flare on return).
    plt.plot(history, label="mean surprise")
    plt.axvspan(ODDBALL_LRANGE, ODDBALL_RRANGE, color="red", alpha=0.2,
                label="oddball (bar omitted)")
    plt.xlabel("frame")
    plt.ylabel("mean surprise")
    plt.title(f"Repetition suppression + mismatch response ({STANDARD})")
    plt.legend()
    plt.savefig("oddball_surprise.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
