import unittest
import pandas as pd
from io import StringIO
from load_data import load_data

class TestLoadData(unittest.TestCase):
    def setUp(self):
        self.csv_data = """col1;col2;col3
1;2;3
4;5;6
7;8;9"""
        self.filepath = StringIO(self.csv_data)

    def test_load_data(self):
        """
        Test the load_data function to ensure it correctly loads data from a file.

        This test compares the DataFrame returned by the load_data function with an
        expected DataFrame to verify that the data is loaded correctly.

        The expected DataFrame contains the following data:
            col1: [1, 4, 7]
            col2: [2, 5, 8]
            col3: [3, 6, 9]

        The test uses pandas' assert_frame_equal function to check if the result
        matches the expected DataFrame.

        Raises:
            AssertionError: If the DataFrame returned by load_data does not match
                            the expected DataFrame.
        """
        expected_data = pd.DataFrame({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col3': [3, 6, 9]
        })
        result_data = load_data(self.filepath)
        pd.testing.assert_frame_equal(result_data, expected_data)

if __name__ == '__main__':
    unittest.main()