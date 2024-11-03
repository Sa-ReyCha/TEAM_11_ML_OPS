import unittest
import pandas as pd
from src.data.split_data import data_split

class TestDataSplit(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'Class': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

    def test_data_split_shapes(self):
        """
        Test the data_split function to ensure that the shapes of the resulting
        training, validation, and test sets are as expected.
        The expected shapes are:
        - X_train: (6, 2)
        - X_val: (2, 2)
        - X_test: (2, 2)
        - y_train: (6,)
        - y_val: (2,)
        - y_test: (2,)
        This test verifies that the data_split function correctly splits the data
        into training, validation, and test sets with the specified shapes.
        """
        X_train, X_test, y_train, y_test, X_val, y_val = data_split(self.data)
        
        self.assertEqual(X_train.shape, (6, 2))
        self.assertEqual(X_val.shape, (2, 2))
        self.assertEqual(X_test.shape, (2, 2))
        self.assertEqual(y_train.shape, (6,))
        self.assertEqual(y_val.shape, (2,))
        self.assertEqual(y_test.shape, (2,))

    def test_data_split_content(self):
        """
        Test the data_split function to ensure that the training, testing, and validation
        sets are correctly split and contain the appropriate data.
        This test checks the following:
        - The training set (X_train) and the test set (X_test) do not share any indices.
        - The training set (X_train) and the validation set (X_val) do not share any indices.
        - The validation set (X_val) and the test set (X_test) do not share any indices.
        Assertions:
        - Asserts that the indices of X_train and X_test are disjoint.
        - Asserts that the indices of X_train and X_val are disjoint.
        - Asserts that the indices of X_val and X_test are disjoint.
        """
        X_train, X_test, y_train, y_test, X_val, y_val = data_split(self.data)
        
        self.assertTrue(set(X_train.index).isdisjoint(X_test.index))
        self.assertTrue(set(X_train.index).isdisjoint(X_val.index))
        self.assertTrue(set(X_val.index).isdisjoint(X_test.index))

if __name__ == '__main__':
    unittest.main()