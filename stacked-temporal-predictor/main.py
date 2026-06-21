"""Live demo: a stacked ConvLSTM watches the camera and paints the prediction error.

The online loop is identical to temporal-predictor/main.py -- measure how wrong the
previous guess was (that error map is the surprise), take one gradient step, predict
again. The only change is the model: StackedPredictor in place of the single-layer
one. There is still a single error signal at the pixel output, so the display is
unchanged: camera input beside the surprise map.

Press 'q' to quit, 'r' to reset the model's memory and watch it re-adapt.
"""

import cv2
import numpy as np
import torch

from stacked import StackedPredictor

SIZE = 64           # frames shrunk to SIZE x SIZE grayscale
LR = 2e-3           # learning rate = how fast surprise fades
GAIN = 4.0          # display brightness for the error map (fixed, so learned scenes go dark)
NUM_LAYERS = 2      # how many ConvLSTM cells are stacked


def to_tensor(gray):
    """HxW uint8 image -> 1x1xHxW float tensor in [0, 1]."""
    return torch.from_numpy(gray).float().div(255.0)[None, None]


def main():
    cam = cv2.VideoCapture(0)
    model = StackedPredictor(in_channels=1, hidden_channels=32, num_layers=NUM_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    pred = None  # our guess for the CURRENT frame, made on the previous iteration
    while True:
        ok, frame = cam.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (SIZE, SIZE))
        x = to_tensor(gray)

        if pred is not None:
            error = x - pred                       # how wrong last step's guess was
            loss = error.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()                             # one online step toward being right
            surprise = error.detach().abs()[0, 0].numpy()
        else:
            surprise = np.zeros((SIZE, SIZE), np.float32)

        model.detach_memory()                      # truncated BPTT, window 1
        pred = model(x)                            # guess the NEXT frame from this one

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
