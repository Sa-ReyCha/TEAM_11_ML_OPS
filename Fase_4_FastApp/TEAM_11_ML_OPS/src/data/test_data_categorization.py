import unittest
import pandas as pd
from data_categorization import atr_list_maker, df_maker, separation_data

class TestDataCategorization(unittest.TestCase):

    def setUp(self):
        """
        Sets up the test environment by creating a DataFrame with predefined data.
        
        The DataFrame contains 54 attributes (Atr1 to Atr54) and a 'Class' column.
        Each attribute and the 'Class' column have three rows of binary values (0 or 1).
        
        Attributes:
            df (pd.DataFrame): The DataFrame containing the test data.
        """
        data = {
            'Atr1': [1, 0, 1],
            'Atr2': [0, 1, 0],
            'Atr3': [1, 1, 0],
            'Atr4': [0, 0, 1],
            'Atr5': [1, 0, 1],
            'Atr6': [0, 1, 0],
            'Atr7': [1, 1, 0],
            'Atr8': [0, 0, 1],
            'Atr9': [1, 0, 1],
            'Atr10': [0, 1, 0],
            'Atr31': [1, 1, 0],
            'Atr32': [0, 0, 1],
            'Atr33': [1, 0, 1],
            'Atr34': [0, 1, 0],
            'Atr35': [1, 1, 0],
            'Atr36': [0, 0, 1],
            'Atr37': [1, 0, 1],
            'Atr38': [0, 1, 0],
            'Atr39': [1, 1, 0],
            'Atr40': [0, 0, 1],
            'Atr41': [1, 0, 1],
            'Atr42': [0, 1, 0],
            'Atr43': [1, 1, 0],
            'Atr44': [0, 0, 1],
            'Atr45': [1, 0, 1],
            'Atr46': [0, 1, 0],
            'Atr47': [1, 1, 0],
            'Atr48': [0, 0, 1],
            'Atr49': [1, 0, 1],
            'Atr50': [0, 1, 0],
            'Atr51': [1, 1, 0],
            'Atr52': [0, 0, 1],
            'Atr53': [1, 0, 1],
            'Atr54': [0, 1, 0],
            'Class': [1, 0, 1]
        }
        self.df = pd.DataFrame(data)

    def test_atr_list_maker(self):
        """
        Test the atr_list_maker function.

        This test checks if the atr_list_maker function correctly transforms a list of integers
        into a list of strings with the prefix 'Atr'.

        Example:
            Input: [1, 2, 3]
            Output: ['Atr1', 'Atr2', 'Atr3']
        """
        result = atr_list_maker([1, 2, 3])
        self.assertEqual(result, ['Atr1', 'Atr2', 'Atr3'])

    def test_df_maker(self):
        """
        Test the df_maker function to ensure it correctly processes the input DataFrame
        and list of columns.

        This test checks that the resulting DataFrame has the expected columns and that
        the 'Class' column contains the correct values.

        Assertions:
            - The columns of the resulting DataFrame should match the expected columns.
            - The 'Class' column in the resulting DataFrame should contain the expected values.
        """
        result = df_maker(self.df, [1, 2, 3])
        expected_columns = ['Atr1', 'Atr2', 'Atr3', 'Class']
        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(result['Class'].tolist(), ['Divorciado', 'Casado', 'Divorciado'])

    def test_separation_data(self):
        """
        Test the separation_data function to ensure it correctly categorizes the data
        into different DataFrames and checks the shape of each resulting DataFrame.

        The function verifies that the result contains the following DataFrames:
        - 'df_Communication_and_Conflict_Management'
        - 'df_Relationship_Harmony_and_Shared_Values'
        - 'df_Emotional_Connection_and_Bonding'
        - 'df_Dissatisfaction_and_Detachment'
        - 'df_Blame_and_Defensiveness'

        Additionally, it checks that each DataFrame has the expected number of columns:
        - 'df_Communication_and_Conflict_Management' should have 20 columns
        - 'df_Relationship_Harmony_and_Shared_Values' should have 13 columns
        - 'df_Emotional_Connection_and_Bonding' should have 14 columns
        - 'df_Dissatisfaction_and_Detachment' should have 4 columns
        - 'df_Blame_and_Defensiveness' should have 7 columns
        """
        result = separation_data(self.df)
        self.assertIn('df_Communication_and_Conflict_Management', result)
        self.assertIn('df_Relationship_Harmony_and_Shared_Values', result)
        self.assertIn('df_Emotional_Connection_and_Bonding', result)
        self.assertIn('df_Dissatisfaction_and_Detachment', result)
        self.assertIn('df_Blame_and_Defensiveness', result)
        self.assertEqual(result['df_Communication_and_Conflict_Management'].shape[1], 20)
        self.assertEqual(result['df_Relationship_Harmony_and_Shared_Values'].shape[1], 13)
        self.assertEqual(result['df_Emotional_Connection_and_Bonding'].shape[1], 14)
        self.assertEqual(result['df_Dissatisfaction_and_Detachment'].shape[1], 4)
        self.assertEqual(result['df_Blame_and_Defensiveness'].shape[1], 7)

if __name__ == '__main__':
    unittest.main()