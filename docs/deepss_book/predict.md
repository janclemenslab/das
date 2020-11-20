# Predict


## GUI
Load audio and DeepSS/Predict:
- model
- postprocessing or events:
For events:
- set the confidence threshold for events (the default of 0.5 is typically good)
- set a minimal distance between events during detection () and segments


Workflow:
1. predict class probabilities using the network
2. cleanup and detect *segment* on- and offset times
3. cleanup and detect *event* times

## Inference parameters
After training, optimize inference parameters:
inference_params = optimize_inference(x, y, model_name)

- choose pre-processing (min_dist for events, fill_gaps, delete_short)
- make precision-recall curve (segments, matched events)
- jitter (events, segment on- and offsets)
- by default, choose threshold that maximizes f1-score (but make this a slider)


## Examples:
### Graphical interface
Load audio and DeepSS/Predict:
- model
- postprocessing or events:
For events:
- set the confidence threshold for events (the default of 0.5 is typically good)
- set a minimal distance between events during detection () and segments


### Command line
`dss-predict file.wav model_trunk`

### Python
- python: `dss.predict.predict('file.wav', model_save_name='model_trunk'))`
- see notebook

## Inference
events, segments = infer(x, model_name, inference_params)   # fall back sensible to defaults (all thres 0.5, no pre-processing) with warning
`events[eventname][seconds/probabilities]`
`segements[segmentname][denselabels/onset_seconds/offset_seconds/probabilities]`

for each class:
type, name, threshold=0.5, min_dist=None, min_len=None, fill_len=None

## Predict class probabilities
```python
class_probabilities = predict(x,  # array [samples, channels] or [samples, frequencies, channels]
       model_name,  # basename of the model
       )
```

## Process segments


## Process events

in a python shell/notebook
```python
import dss.predict
song = scipy.io.wavefile.read('filename')
segment_probabilities, segment_labels, event_times, event_confidence = dss.predict(song, modelfilename)
```
