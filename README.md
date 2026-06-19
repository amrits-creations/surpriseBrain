# Surprise Vision

A tiny, watchable demo of one idea: **perception is mostly prediction, and
attention is mostly surprise.** A small model watches a video stream, constantly
guesses the *next* frame, and paints a **surprise map** — not the picture itself,
but a picture of *how wrong its guesses are*, pixel by pixel. **Dark** = "I
expected this" (anything already learned fades to black, even while moving);
**bright** = "I did not see this coming" (novel events flare, then fade on their
own as the model learns them).

## temporal-predictor *(built)*

A single-layer **ConvLSTM** watches the live camera and learns *while* watching:
each frame it measures how wrong its last guess was (that error map **is** the
surprise), takes one online gradient step, then guesses the next frame. Motion it
has seen becomes predictable and fades to black on its own — only *failed*
predictions stay bright. The **oddball experiment** turns that "it fades" vibe
into a measurement: feed a repeating synthetic stimulus until surprise flattens,
then break the pattern once, and the logged curve shows both a decaying
repetition-suppression envelope and a one-off mismatch spike — the classic oddball
paradigm.

```bash
.venv/bin/python temporal-predictor/main.py   # q to quit, r to reset memory
```

- **Code:** [main.py](temporal-predictor/main.py) (live loop),
  [oddball.py](temporal-predictor/oddball.py) (the experiment)
- **Report:** [temporal-predictor-report.md](temporal-predictor/temporal-predictor-report.md)
</content>
