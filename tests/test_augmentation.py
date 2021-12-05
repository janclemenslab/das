import das.augmentation
import numpy as np
import das


def test_param():
    das.augmentation.Constant(10)((3,2))
    das.augmentation.Normal(1, 2)((3,2))
    das.augmentation.Uniform(1, 2)((3,2))

def test_augmentation():
    signal = np.arange(0, 100) / 100

    aug_gain = das.augmentation.Gain(gain=das.augmentation.Uniform(-1, 1))

    aug_offset = das.augmentation.Offset(offset=das.augmentation.Uniform(-1, 1))

    aug_addnoise = das.augmentation.MaskNoise(mean=das.augmentation.Uniform(-1, 1),
                                            std=das.augmentation.Uniform(0, 0.1))

    aug_masknoise = das.augmentation.MaskNoise(duration=das.augmentation.Uniform(5, 20),
                                            std=das.augmentation.Uniform(0, 0.1))

    aug_masknoise = das.augmentation.MaskNoise(duration=das.augmentation.Uniform(5, 20),
                                            std=das.augmentation.Uniform(0, 0.1),
                                            add=False)

    aug_horzflip = das.augmentation.HorizontalFlip(flip=das.augmentation.Uniform(-1, 1))

    aug_upsamp = das.augmentation.Upsampling(factor=das.augmentation.Uniform(1, 1.5))

    aug_circshift = das.augmentation.CircShift(shift=das.augmentation.Uniform(-10, 10))

def test_augmentations():
    aug_circshift = das.augmentation.CircShift(shift=das.augmentation.Uniform(-10, 10))
    aug_masknoise = das.augmentation.MaskNoise(duration=das.augmentation.Uniform(5, 20),
                                            std=das.augmentation.Uniform(0, 0.1))
    augs = [aug_circshift, aug_masknoise]
    aug_shiftnoise = das.augmentation.Augmentations(augs)
