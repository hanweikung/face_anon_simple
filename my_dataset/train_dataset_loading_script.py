# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import os

import datasets
import pandas as pd


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
_URLS = {
    "metadata_path": "/path/to/face_anon_simple/my_dataset/train/train.jsonl",
    "images_dir": "/path/to/face_anon_simple/my_dataset/train/",
}


# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        features = datasets.Features(
            {
                "source_image": datasets.Image(),
                "conditioning_image": datasets.Image(),
                "ground_truth": datasets.Image(),
                "source_image_path": datasets.Value("string"),
                "conditioning_image_path": datasets.Value("string"),
                "ground_truth_path": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name
        metadata_path = _URLS["metadata_path"]
        images_dir = _URLS["images_dir"]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, metadata_path, images_dir):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            source_image_path = row["source_image"]
            source_image_path = os.path.join(images_dir, source_image_path)
            source_image = open(source_image_path, "rb").read()

            conditioning_image_path = row["conditioning_image"]
            conditioning_image_path = os.path.join(images_dir, conditioning_image_path)
            conditioning_image = open(conditioning_image_path, "rb").read()

            ground_truth_path = row["ground_truth"]
            ground_truth_path = os.path.join(images_dir, ground_truth_path)
            ground_truth = open(ground_truth_path, "rb").read()

            yield (
                "-".join([source_image_path, conditioning_image_path]),
                {
                    "source_image": {
                        "path": source_image_path,
                        "bytes": source_image,
                    },
                    "conditioning_image": {
                        "path": conditioning_image_path,
                        "bytes": conditioning_image,
                    },
                    "ground_truth": {
                        "path": ground_truth_path,
                        "bytes": ground_truth,
                    },
                    "source_image_path": source_image_path,
                    "conditioning_image_path": conditioning_image_path,
                    "ground_truth_path": ground_truth_path,
                },
            )
