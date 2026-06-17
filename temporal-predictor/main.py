"""Live demo: watch the camera, predict the next frame, paint the surprise.

The loop is where "repetition suppression" actually comes from. Each frame we:
  1. measure how wrong our previous guess was  -> that error IS the surprise map,
  2. take one gradient step to be less wrong next time (online learning),
  3. make a fresh guess of the upcoming frame.

Because the model keeps learning while it watches, motion it has seen before
becomes predictable and its error shrinks toward zero on its own -> it fades to
black, even while still moving. Only motion it *failed* to predict stays bright.

Press 'q' to quit, 'r' to reset the model's memory and watch it re-adapt.
"""

import cv2
import numpy as np
import torch

from convlstm import TemporalPredictor

SIZE = 64       # frames shrunk to SIZE x SIZE grayscale. Smaller = faster (CPU real-time).
LR = 2e-3       # learning rate = how fast surprise fades. This is the main "feel" knob.    Note: I doubled it to speed up how long it takes to "read the room" and stabilize it's weights from random values.
GAIN = 4.0      # display brightness for the error map. Fixed (not per-frame) on purpose,
                # so a learned/static scene genuinely goes dark instead of self-normalizing.


def to_tensor(gray):
    """HxW uint8 image -> 1x1xHxW float tensor in [0, 1]."""
    return torch.from_numpy(gray).float().div(255.0)[None, None]


def main():
    cam = cv2.VideoCapture(0)
    model = TemporalPredictor(in_channels=1, hidden_channels=32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    pred = None  # our guess for the CURRENT frame, made on the previous iteration
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SIZE, SIZE))
        x = to_tensor(gray) # FRAME HAS BEEN CONVERTED TO X.

        if pred is not None:
            error = x - pred                       # how wrong last step's guess was
            loss = error.pow(2).mean()             # train on that mistake...
            opt.zero_grad()
            loss.backward()
            opt.step()                             # ...one online step toward being right
            surprise = error.detach().abs()[0, 0].numpy()
        else:
            surprise = np.zeros((SIZE, SIZE), np.float32)

        # Cut the memory loose from the old graph so backprop only ever spans one
        # step (truncated BPTT, window 1) and the graph doesn't grow without bound.
        if model.h is not None and model.c is not None:
            model.h, model.c = model.h.detach(), model.c.detach()

        pred = model(x)                            # guess the NEXT frame from this one

        # Paint: surprise map beside the (resized) camera input so the demo is legible.
        smap = np.clip(surprise * GAIN * 255, 0, 255).astype(np.uint8)
        view = np.hstack([cv2.resize(gray, (480, 480), interpolation=cv2.INTER_NEAREST),
                          cv2.resize(smap, (480, 480), interpolation=cv2.INTER_NEAREST)])
        cv2.imshow("camera  |  surprise", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            model.reset()
            pred = None

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
