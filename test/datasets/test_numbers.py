import pytest
from src.datasets import numbers


def test_numbers():
    dataset = numbers(subset="training", path="./data/numbers/train")

    image_width = 32
    image_height = 32
    number_of_channels = 3
    batch_size = 32
    num_classes = 10
    # check if the first batch has the correct shape - image size is 32x32, batch size is 32 and there are 3 channels
    for batch in dataset.take(1):
        assert batch[0].shape == (
            batch_size,
            image_width,
            image_height,
            number_of_channels,
        )
        assert batch[1].shape == (batch_size, num_classes)
