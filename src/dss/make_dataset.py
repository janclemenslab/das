import numpy as np
import zarr


def init_store(nb_channels, nb_classes, samplerate=None,
               class_names=None, class_types=None,
               store_type=zarr.DictStore, store_name='store.zarr', chunk_len=1_000_000):
    """[summary]

    Args:
        nb_channels ([type]): [description]
        nb_classes ([type]): [description]
        samplerate ([type], optional): [description]. Defaults to None.
        class_names ([type], optional): [description]. Defaults to None.
        class_types ([type], optional): [description]. Defaults to None.
        store_type ([type], optional): [description]. Defaults to zarr.DictStore.
        store_name (str, optional): [description]. Defaults to 'store.zarr'.
        chunk_len ([type], optional): [description]. Defaults to 1_000_000.

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    if class_names is not None and nb_classes is not None and len(class_names) != nb_classes:
        raise ValueError(f'Number of classes ({nb_classes}) needs to match len(class_names) ({len(class_names)}).')
    if class_types is not None and nb_classes is not None and len(class_names) != nb_classes:
        raise ValueError(f'Number of classes ({nb_classes}) needs to match len(class_names) ({len(class_types)}).')

    # initialize the store
    store = store_type(store_name)
    root = zarr.group(store=store, overwrite=True)  # need to def the root
    for target in ['train', 'val', 'test']:
        root.empty(name=f'{target}/x', shape=(0, nb_channels), chunks=(chunk_len, nb_channels), dtype=np.float16)
        root.empty(name=f'{target}/y', shape=(0, nb_classes), chunks=(chunk_len, nb_classes), dtype=np.float16)
        root.empty(name=f'{target}/eventtimes', shape=(0, nb_classes), chunks=(1_000,), dtype=np.float)

    # init metadata - since attrs cannot be appended to, we init a dict here, populate it with information below and finaly assign it to root.attrs
    root.attrs['samplerate_x_Hz'] = samplerate
    root.attrs['samplerate_y_Hz'] = samplerate
    root.attrs['eventtimes_units'] = 'seconds'
    root.attrs['class_names'] = class_names
    root.attrs['class_types'] = class_types

    for target in ['train', 'val', 'test']:
        root.attrs[f'filename_startsample_{target}'] = []
        root.attrs[f'filename_endsample_{target}'] = []
        root.attrs[f'filename_{target}'] = []
    return root


def events_to_probabilities(eventsamples, desired_len=None, extent=61):
    """Converts list of events to one-hot-encoded probability vectors.

    Args:
        eventsamples (List[int]): List of event "times" in samples.
        desired_len (float, optional): Length of the probability vector.
                                       Events exceeding `desired_len` will be ignored.
                                       Defaults to `max(eventsamples) + extent`.
        extent (int, optional): Temporal extent of an event in the probability vector.
                                Each event will be represented as a box with a duration `exent` samples centered on the event.
                                Defaults to 61 samples (+/-30 samples).
    Returns:
        probabilities: np.array with shape [desired_len, 2]
                       where `probabilities[:, 0]` corresponds to the probability of no event
                       and `probabilities[:, 0]` corresponds to the probability of an event.
    """
    if desired_len is None:
        desired_len = max(eventsamples) + extent
    else:
        eventsamples = eventsamples[eventsamples < desired_len - extent]  # delete all eventsamples exceeding desired_len
    probabilities = np.zeros((desired_len, 2))
    probabilities[eventsamples, 1] = 1
    probabilities[:, 1] = np.convolve(probabilities[:, 1], np.ones((extent,)), mode='same')
    probabilities[:, 0] = 1 - probabilities[:, 1]
    return probabilities
