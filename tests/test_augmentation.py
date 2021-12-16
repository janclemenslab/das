import das.augmentation
import numpy as np
import das
import yaml


def test_param():
    das.augmentation.Constant(10)((3,2))
    das.augmentation.Normal(1, 2)((3,2))
    das.augmentation.Uniform(1, 2)((3,2))


def test_augmentation():
    signal = np.arange(0, 100) / 100
    signal = signal[np.newaxis, :, np.newaxis]

    aug_gain = das.augmentation.Gain(gain=das.augmentation.Uniform(-1, 1))
    aug_gain(signal)

    aug_offset = das.augmentation.Offset(offset=das.augmentation.Uniform(-1, 1))
    aug_offset(signal)

    aug_addnoise = das.augmentation.MaskNoise(mean=das.augmentation.Uniform(-1, 1),
                                            std=das.augmentation.Uniform(0, 0.1))
    aug_addnoise(signal)

    aug_masknoise = das.augmentation.MaskNoise(duration=das.augmentation.Uniform(5, 20),
                                            std=das.augmentation.Uniform(0, 0.1))
    aug_masknoise(signal)

    aug_masknoise = das.augmentation.MaskNoise(duration=das.augmentation.Uniform(5, 20),
                                            std=das.augmentation.Uniform(0, 0.1),
                                            add=False)
    aug_masknoise(signal)

    aug_horzflip = das.augmentation.HorizontalFlip(flip=das.augmentation.Uniform(-1, 1))
    aug_horzflip(signal)

    aug_upsamp = das.augmentation.Upsampling(factor=das.augmentation.Uniform(1, 1.5))
    aug_upsamp(signal)

    aug_circshift = das.augmentation.CircShift(shift=das.augmentation.Uniform(-10, 10))
    aug_circshift(signal)


def test_augmentations():
    aug_circshift = das.augmentation.CircShift(shift=das.augmentation.Uniform(-10, 10))
    aug_masknoise = das.augmentation.MaskNoise(duration=das.augmentation.Uniform(5, 20),
                                            std=das.augmentation.Uniform(0, 0.1))
    augs = [aug_circshift, aug_masknoise]
    aug_shiftnoise = das.augmentation.Augmentations(augs)

    signal = np.arange(0, 100) / 100
    signal = signal[np.newaxis, :, np.newaxis]
    aug_shiftnoise(signal)

def test_from_yaml():
    yaml_string ="""
    Gain:
      gain:
        Uniform:
          lower: 0.8
          upper: 1.25

    MaskNoise:
      std:
        Normal:
          mean: 0.1
          std: 1
        mean:
        Constant:
          value: 0

    HorizontalFlip:
      flip:
        Uniform:
        """
    aug_dict = yaml.safe_load(yaml_string)
    augs = das.augmentation.Augmentations.from_dict(aug_dict)

    signal = np.arange(0, 100) / 100
    signal = signal[np.newaxis, :, np.newaxis]
    augs(signal)
    assert len(augs) == 3
