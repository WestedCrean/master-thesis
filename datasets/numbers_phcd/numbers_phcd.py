"""numbers_phcd dataset."""

import tensorflow_datasets as tfds
import pathlib
# TODO(numbers_phcd): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
PHCD handwritten characters dataset - numbers only
"""

# TODO(numbers_phcd): BibTeX citation
_CITATION = """
"""


class NumbersPhcd(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for numbers_phcd dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(numbers_phcd): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=['no', 'yes']),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(numbers_phcd): Downloads the data and defines the splits
    path = pathlib.Path("../data/numbers").resolve()

    # TODO(numbers_phcd): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train'),
        'test': self._generate_examples(path / 'test'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(numbers_phcd): Yields (key, example) tuples from the dataset
    for f in path.glob('**/*.png'):
      yield f.name, {
          'image': f,
          'label': f.parent.name,
      }
