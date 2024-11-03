import unittest
import numpy as np
from data_scale import scale_data

class TestScaleData(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[1, 2], [3, 4], [5, 6]])
        self.X_val = np.array([[1, 2], [3, 4]])
        self.X_test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    def test_scale_data_shapes(self):
        """
        Test the scale_data function to ensure that the shapes of the scaled data
        match the shapes of the original data.

        This test verifies that the scale_data function does not alter the shape
        of the input datasets (X_train, X_val, X_test) after scaling.

        Asserts:
            The shape of X_train_scaled is equal to the shape of X_train.
            The shape of X_val_scaled is equal to the shape of X_val.
            The shape of X_test_scaled is equal to the shape of X_test.
        """
        X_train_scaled, X_val_scaled, X_test_scaled = scale_data(self.X_train, self.X_val, self.X_test)
        self.assertEqual(X_train_scaled.shape, self.X_train.shape)
        self.assertEqual(X_val_scaled.shape, self.X_val.shape)
        self.assertEqual(X_test_scaled.shape, self.X_test.shape)

    def test_scale_data_mean(self):
        """
        Test the scale_data function to ensure that the mean of the scaled training data is approximately zero.

        This test checks if the mean of each feature in the scaled training data (X_train_scaled) is close to zero
        after applying the scale_data function. The tolerance for the mean value is set to 1e-7.

        Asserts:
            The mean of each feature in X_train_scaled is approximately zero.
        """
        X_train_scaled, X_val_scaled, X_test_scaled = scale_data(self.X_train, self.X_val, self.X_test)
        self.assertTrue(np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7))

    def test_scale_data_std(self):
        """
        Test the scale_data function to ensure that the standard deviation of the 
        scaled training data is approximately 1.

        This test checks if the scale_data function correctly scales the training 
        data (X_train) such that its standard deviation along each feature axis is 
        close to 1, within a tolerance of 1e-7.

        Asserts:
            The standard deviation of the scaled training data (X_train_scaled) 
            along each feature axis is approximately 1.
        """
        X_train_scaled, X_val_scaled, X_test_scaled = scale_data(self.X_train, self.X_val, self.X_test)
        self.assertTrue(np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7))

if __name__ == '__main__':
    unittest.main()