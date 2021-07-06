"""imagenet_21k_p_winter dataset."""

import tensorflow_datasets as tfds
import os

_DESCRIPTION = """
ImageNet-21K processed by cleaning invalid classes, allocating a validation split and resizing images to 224x224.
This is the winter 21 version provided by image-net.org and processed in the same manner as in the paper.
Winter21 ImageNet-21K-P (this version) contains 10450 classes, where the trainset has 11060223 samples and the test set has 522500 samples.
"""

_CITATION = """
@misc{ridnik2021imagenet21k,
      title={ImageNet-21K Pretraining for the Masses}, 
      author={Tal Ridnik and Emanuel Ben-Baruch and Asaf Noy and Lihi Zelnik-Manor},
      year={2021},
      eprint={2104.10972},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

_LABELS_FNAME = 'imagenet_21k_p_winter_labels.txt'


class Imagenet21kPWinter(tfds.core.GeneratorBasedBuilder):
    """ImageNet-21K P Winter, AKA Winter21 ImageNet-21K Processed"""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  manual_dir should contain one file: imagenet21k_resized.tar.gz.
  You need to register on http://www.image-net.org/download-images in order
  to get the link to download the dataset.
  """

    def _info(self) -> tfds.core.DatasetInfo:
        names_file = tfds.core.tfds_path(_LABELS_FNAME)
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image':
                    tfds.features.Image(shape=(224, 224, 3),
                                        encoding_format='jpeg'),
                'label':
                    tfds.features.ClassLabel(names_file=names_file),
                'file_name':
                    tfds.features.Text(),
            }),
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage='https://github.com/Alibaba-MIIL/ImageNet21K',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = os.path.join(dl_manager.manual_dir, 'imagenet21k_resized.tar.gz')
        return [
            tfds.core.SplitGenerator(name=tfds.Split.TRAIN,
                                     gen_kwargs={
                                         "archive_path": path,
                                         "is_validation": False
                                     }),
            tfds.core.SplitGenerator(name=tfds.Split.VALIDATION,
                                     gen_kwargs={
                                         "archive_path": path,
                                         "is_validation": True
                                     }),
        ]

    def _generate_examples(self, archive_path, is_validation):
        """Yields examples."""
        if is_validation:
            for fpath, fobj in tfds.download.iter_archive(
                    archive_path, tfds.download.ExtractMethod.TAR):
                fpath_split = fpath.split(os.path.sep)
                if 'imagenet21k_val' in fpath_split and fpath_split[-1].lower(
                ).endswith('.jpeg'):
                    record = {
                        'image': fobj,
                        'label': fpath_split[-2],
                        'file_name': fpath_split[-1]
                    }
                    yield fpath_split[-1], record
        else:
            for fpath, fobj in tfds.download.iter_archive(
                    archive_path, tfds.download.ExtractMethod.TAR):
                fpath_split = fpath.split(os.path.sep)
                if 'imagenet21k_train' in fpath_split and fpath_split[-1].lower(
                ).endswith('.jpeg'):
                    record = {
                        'image': fobj,
                        'label': fpath_split[-2],
                        'file_name': fpath_split[-1]
                    }
                    yield fpath_split[-1], record
