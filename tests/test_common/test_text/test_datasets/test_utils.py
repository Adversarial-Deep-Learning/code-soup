import unittest

from parameterized import parameterized_class

from code_soup.common.text.datasets.utils import dataset_mapping


@parameterized_class(
    ("x", "expected_output"),
    [
        (
            {"sentence": "Chuffed to bits!", "label": 0.598},
            {"x": "Chuffed to bits!", "y": 1},
        ),
        ({"sentence": "Hello", "label": 0.342}, {"x": "Hello", "y": 0}),
    ],
)
class TestTextDatasetUtilsDatasetMapping(unittest.TestCase):
    """
    Parameterized test cases for the common/text/datasets/utils/dataset_mapping
    function.

    Args: ("x", "expected_output")
    """

    def setUp(self):
        pass

    def test_output(self):
        self.assertDictEqual(dataset_mapping(self.x), self.expected_output)
