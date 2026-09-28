# GUI

Run `das` or `das gui` to open the xarray-behave annotation window. Open one audio file and its annotation CSV at a time. The DAS GUI has no recording-folder or project navigation; training can still read a folder of WAV files.

Use **DAS → Train** to select a WAV folder or an existing DAS training dataset, choose the model, and set training options. The dialog includes the segment hysteresis thresholds, event threshold and spacing, and manual postprocessing controls. WhisperSeg training needs a local DAS `.ckpt` initial model.

Use **DAS → Predict** to select a DAS `.ckpt` or a legacy DAS H5/YAML model. The dialog exposes the same detection and postprocessing controls, plus evaluation options. Predictions appear in the current annotation view for review and can be saved as CSV.

Choose **File → Open etho folder** to open ethodrome recordings; use **File → Save dataset** to save one in Zarr format. **DAS → Make dataset for training** builds a train/validation/test `.npy` dataset from an annotated audio folder. **File → Load dataset** opens an existing dataset.
