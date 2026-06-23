import os
import pandas as pd
import unittest

class TestCustomerSegmentationPipeline(unittest.TestCase):
    
    def test_output_file_exists(self):
        """Check if the final segmented CSV was created."""
        self.assertTrue(os.path.exists("outputs/segmented_customers.csv"))
        
    def test_output_schema(self):
        """Check if the output CSV has the correct columns."""
        df = pd.read_csv("outputs/segmented_customers.csv")
        expected_cols = ["CustomerID", "cluster", "segment", "recency", "frequency", "monetary"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_model_files_exist(self):
        """Check if the model and scaler were saved."""
        self.assertTrue(os.path.exists("models/kmeans_model.pkl"))
        self.assertTrue(os.path.exists("models/scaler.pkl"))

    def test_segment_count(self):
        """Check if there are exactly 4 segments created."""
        df = pd.read_csv("outputs/segmented_customers.csv")
        self.assertEqual(df["segment"].nunique(), 4)

if __name__ == "__main__":
    unittest.main()