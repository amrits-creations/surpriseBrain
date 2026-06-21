"""Oddball experiment for the stacked predictor.

Same synthetic stimulus, online loop, and logged surprise curve as
temporal-predictor/oddball.py; only the model differs (StackedPredictor). Running it
with the same parameters lets the stacked model's repetition-suppression curve and
mismatch response be compared directly against the single-layer numbers.

A standard stimulus is presented until the surprise settles, then omitted for a range
of frames (the deviant). Set STANDARD below:
  - "static": a bar held still. "next = current" is learnable exactly, so surprise
    fades to true black.
  - "moving": a bar swept across the frame. A one-step-ahead predictor cannot place a
    sharp moving edge exactly, so a faint rim of error survives.

Press 'q' to stop early. On exit the surprise-per-frame curve is plotted and saved.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from stacked import StackedPredictor

SIZE = 64               # frame is SIZE x SIZE grayscale
LR = 5e-4               # learning rate (matches main.py and the single-layer report)
GAIN = 4.0              # display brightness for the error map
NUM_LAYERS = 2          # how many ConvLSTM cells are stacked

STANDARD = "moving"     # "static" -> fades to true black; "moving" -> fades to dark grey
BAR = 8                 # width of the bar, in pixels
STEP = 1                # how far the bar moves each frame (only used when STANDARD == "moving")
STATIC_X = 28           # the bar's fixed column (only used when STANDARD == "static")
TOTAL = 1000            # how many frames the experiment runs for
ODDBALL_LRANGE = 800    # first frame of the deviant (the bar is omitted from here...)
ODDBALL_RRANGE = 820    # ...to here, inclusive


def to_tensor(gray):
    """HxW uint8 image -> 1x1xHxW float tensor in [0, 1]."""
    return torch.from_numpy(gray).float().div(255.0)[None, None]


def stimulus(t):
    """The frame shown at step t: a white bar, omitted during the oddball range."""
    frame = np.zeros((SIZE, SIZE), np.uint8)
    if ODDBALL_LRANGE <= t <= ODDBALL_RRANGE:
        return frame                                  # deviant: expected bar is missing
    x = STATIC_X if STANDARD == "static" else (t * STEP) % SIZE
    for k in range(BAR):
        frame[:, (x + k) % SIZE] = 255                # draw the bar, wrapping the edge cleanly
    return frame


def main():
    model = StackedPredictor(in_channels=1, hidden_channels=32, num_layers=NUM_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = []        # mean surprise per frame -- the measurement
    pred = None         # our guess for the CURRENT frame, made on the previous iteration
    for t in range(TOTAL):
        gray = stimulus(t)
        x = to_tensor(gray)

        if pred is not None:
            error = x - pred                           # how wrong last step's guess was
            loss = error.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()                                 # one online step toward being right
            surprise = error.detach().abs()[0, 0].numpy()
        else:
            surprise = np.zeros((SIZE, SIZE), np.float32)
        history.append(float(surprise.mean()))

        model.detach_memory()                          # truncated BPTT, window 1
        pred = model(x)                                # guess the NEXT frame from this one

        smap = np.clip(surprise * GAIN * 255, 0, 255).astype(np.uint8)
        view = np.hstack([cv2.resize(gray, (480, 480), interpolation=cv2.INTER_NEAREST),
                          cv2.resize(smap, (480, 480), interpolation=cv2.INTER_NEAREST)])
        cv2.imshow("stimulus  |  surprise", view)
        if cv2.waitKey(40) & 0xFF == ord('q'):         # ~25 fps so the motion is watchable
            break

    cv2.destroyAllWindows()

    plt.plot(history, label="mean surprise")
    plt.axvspan(ODDBALL_LRANGE, ODDBALL_RRANGE, color="red", alpha=0.2,
                label="oddball (bar omitted)")
    plt.xlabel("frame")
    plt.ylabel("mean surprise")
    plt.title(f"Stacked predictor: repetition suppression + mismatch ({STANDARD})")
    plt.legend()
    plt.savefig("oddball_surprise.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
