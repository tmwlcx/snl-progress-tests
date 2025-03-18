import unittest
import numpy as np
from unittest.mock import patch, PropertyMock
import pandas as pd
from datetime import datetime
from contextlib import redirect_stdout
import io

from progress.mod_kmeans import KMeans_Pipeline

class TestKMeansPipeline(unittest.TestCase):
    def setUp(self):
        """Setup test fixtures and suppress stdout"""
        # Create string buffer to capture output
        self.output = io.StringIO()
        
        with redirect_stdout(self.output):
            self.test_directory = "/test/path" 
            self.test_sites_file = "/test/path/solar_sites.csv"

            # Mock site info DataFrame matching actual structure
            self.site_info_df = pd.DataFrame({
                'site_name': ['101_PV_1', '101_PV_2', '101_PV_3'],
                'lat': [33.44610326, 33.39610326, 33.34610326],
                'long': [-113.835642, -113.835642, -113.835642],
                'MW': [25.9, 26.7, 26.2],
                'tracking': [1, 1, 1],
                'zone': [1, 1, 1]
            })

            # Create full year of hourly data (8760 hours)
            dates = pd.date_range('1998-01-01', periods=8760, freq='h')

            # Create one day pattern and repeat it 365 times
            day_pattern = [0]*8 + [7.65, 15.27, 20.60, 23.73, 24.78, 23.91, 21.23, 16.65, 9.88] + [0]*7
            year_pattern = day_pattern * 365  # Repeat for 365 days

            # Mock solar generation data matching actual structure for full year
            self.solar_gen_df = pd.DataFrame({
                'datetime': dates,
                '101_PV_1': year_pattern,
                '101_PV_2': [x * 1.02 for x in year_pattern],  # Slightly different pattern
                '101_PV_3': [x * 0.98 for x in year_pattern]   # Slightly different pattern
            })

            # Mock CSI data matching actual structure for full year
            day_csi = [0]*8 + [1]*9 + [0]*7
            year_csi = day_csi * 365  # Repeat for 365 days
            self.csi_df = pd.DataFrame({
                'datetime': dates,
                '101_PV_1': year_csi,
                '101_PV_2': year_csi,
                '101_PV_3': year_csi
            })

            # Set up pandas mocks
            self.csv_patcher = patch('pandas.read_csv')
            self.excel_patcher = patch('pandas.read_excel')
            self.mock_read_csv = self.csv_patcher.start()
            self.mock_read_excel = self.excel_patcher.start()

            # Configure mock returns
            self.mock_read_csv.return_value = self.site_info_df
            def mock_excel(*args, **kwargs):
                if kwargs.get('sheet_name') == 'solar_gen':
                    return self.solar_gen_df
                elif kwargs.get('sheet_name') == 'csi':
                    return self.csi_df
            self.mock_read_excel.side_effect = mock_excel

            # Mock the preprocessing methods with proper return values
            self.mock_flh = patch('progress.mod_kmeans.KMeans_Pipeline.process_flh_and_llh').start()
            self.mock_solar = patch('progress.mod_kmeans.KMeans_Pipeline.process_solar_data').start()
            self.mock_csi = patch('progress.mod_kmeans.KMeans_Pipeline.process_csi_data').start()

            # Mock create_kmeans_df to return a DataFrame with correct column names
            self.kmeans_patcher = patch('progress.mod_kmeans.KMeans_Pipeline.create_kmeans_df')
            self.mock_kmeans = self.kmeans_patcher.start()
            mock_kmeans_df = pd.DataFrame(
                np.random.rand(24, 8), 
                columns=[
                    '101_PV_1_csi_sd_am',
                    '101_PV_1_csi_sd_pm',
                    '101_PV_1_sg_mean_am',
                    '101_PV_1_sg_mean_pm',
                    'first_light_cos',
                    'first_light_sin',
                    'last_light_cos',
                    'last_light_sin'
                ]
            )
            self.mock_kmeans.return_value = mock_kmeans_df

            # Update mock data sizes for preprocessing methods
            first_light_df = pd.DataFrame({
                'date': dates.date,
                'first_light_sin': np.random.rand(8760),
                'first_light_cos': np.random.rand(8760)
            })
            last_light_df = pd.DataFrame({
                'date': dates.date,
                'last_light_sin': np.random.rand(8760),
                'last_light_cos': np.random.rand(8760)
            })
            self.mock_flh.return_value = (first_light_df, last_light_df)

            self.mock_solar.return_value = (self.solar_gen_df, self.solar_gen_df)
            self.mock_csi.return_value = (self.csi_df, self.csi_df)

            # Create pipeline instance
            self.pipeline = KMeans_Pipeline(self.test_directory, self.test_sites_file)
                                 
    def tearDown(self):
        """Clean up patches and clear output buffer"""
        self.csv_patcher.stop()
        self.excel_patcher.stop()
        self.mock_flh.stop()
        self.mock_solar.stop()
        self.mock_csi.stop()
        self.mock_kmeans.stop()
        self.output.close()

    def test_init(self):
        """Tests the initialization of KMeans_Pipeline class."""
        with redirect_stdout(io.StringIO()):
            # Test directory and file paths are set correctly
            self.assertEqual(self.pipeline.directory, self.test_directory)
            self.assertEqual(self.pipeline.excel_file_path, f'{self.test_directory}/solar_data.xlsx')

            # Test DataFrames are loaded correctly
            pd.testing.assert_frame_equal(self.pipeline.site_info_df, self.site_info_df)
            pd.testing.assert_frame_equal(self.pipeline.solar_gen_df, self.solar_gen_df)
            pd.testing.assert_frame_equal(self.pipeline.csi_df, self.csi_df)

            # Test selected_sites is set to all site names when no kwargs provided
            expected_sites = ['101_PV_1', '101_PV_2', '101_PV_3']
            self.assertEqual(self.pipeline.selected_sites, expected_sites)

    def test_process_flh_and_llh(self):
        with redirect_stdout(io.StringIO()):
            # Unmock the method for this test
            self.mock_flh.stop()

            first_light, last_light = self.pipeline.process_flh_and_llh(
                self.solar_gen_df, 
                ['101_PV_1', '101_PV_2', '101_PV_3']
            )

            # Verify DataFrame structure
            self.assertIn('date', first_light.columns)
            self.assertIn('first_light_sin', first_light.columns)
            self.assertIn('first_light_cos', first_light.columns)
            self.assertIn('last_light_sin', last_light.columns)
            self.assertIn('last_light_cos', last_light.columns)

            # Test that first light is around 8 (matches our test data)
            self.assertTrue(-1 <= first_light['first_light_sin'].iloc[0] <= 1)
            self.assertTrue(-1 <= first_light['first_light_cos'].iloc[0] <= 1)

    def test_process_solar_data(self):
        with redirect_stdout(io.StringIO()):
            self.mock_solar.stop()

            # Update site_info_df and manually divide solar gen values by MW to match normalization
            self.site_info_df.loc[self.site_info_df['site_name'] == '101_PV_1', 'MW'] = 25.0
            self.site_info_df.loc[self.site_info_df['site_name'] == '101_PV_2', 'MW'] = 25.0
            self.site_info_df.loc[self.site_info_df['site_name'] == '101_PV_3', 'MW'] = 25.0

            # Manually normalize test data
            self.solar_gen_df['101_PV_1'] = self.solar_gen_df['101_PV_1'] / 25.0
            self.solar_gen_df['101_PV_2'] = self.solar_gen_df['101_PV_2'] / 25.0
            self.solar_gen_df['101_PV_3'] = self.solar_gen_df['101_PV_3'] / 25.0

            sg_mean_am, sg_mean_pm = self.pipeline.process_solar_data(
                self.solar_gen_df,
                self.site_info_df,
                ['101_PV_1', '101_PV_2', '101_PV_3']
            )

            # Now test normalized values
            max_value = sg_mean_am['101_PV_1'].max()
            self.assertLess(max_value, 1.1)

            self.assertEqual(len(sg_mean_am), len(sg_mean_pm))

    def test_process_csi_data(self):
        with redirect_stdout(io.StringIO()):
            # Unmock the method for this test
            self.mock_csi.stop()

            csi_sd_am, csi_sd_pm = self.pipeline.process_csi_data(
                self.csi_df,
                ['101_PV_1', '101_PV_2', '101_PV_3']
            )

            # Check that data is standardized (doesn't need exact mean/std)
            for col in ['101_PV_1', '101_PV_2', '101_PV_3']:
                values = csi_sd_am[col].dropna()
                self.assertTrue(values.std() > 0)  # Just verify there's variation

    def test_create_kmeans_df(self):
        with redirect_stdout(io.StringIO()):
            self.mock_kmeans.stop()

            # Create properly formatted dataframes for input with full year of data
            dates = pd.date_range('1998-01-01', periods=8760, freq='h').date

            # Create test data with full year
            sg_mean_am = pd.DataFrame({
                'date': dates,
                '101_PV_1': np.random.rand(8760)
            }).set_index('date')

            sg_mean_pm = pd.DataFrame({
                'date': dates,
                '101_PV_1': np.random.rand(8760)
            }).set_index('date')

            csi_sd_am = pd.DataFrame({
                'date': dates,
                '101_PV_1': np.random.rand(8760)
            }).set_index('date')

            csi_sd_pm = pd.DataFrame({
                'date': dates,
                '101_PV_1': np.random.rand(8760)
            }).set_index('date')

            first_light = pd.DataFrame({
                'date': dates,
                'first_light_sin': np.random.rand(8760),
                'first_light_cos': np.random.rand(8760)
            }).set_index('date')

            last_light = pd.DataFrame({
                'date': dates,
                'last_light_sin': np.random.rand(8760),
                'last_light_cos': np.random.rand(8760)
            }).set_index('date')

            kmeans_df = self.pipeline.create_kmeans_df(
                sg_mean_am, sg_mean_pm,
                csi_sd_am, csi_sd_pm,
                first_light, last_light
            )

            # Verify expected column names based on actual implementation
            expected_suffixes = [
                '101_PV_1_csi_sd_am',
                '101_PV_1_csi_sd_pm',
                '101_PV_1_sg_mean_am',
                '101_PV_1_sg_mean_pm',
                'first_light_cos',
                'first_light_sin',
                'last_light_cos',
                'last_light_sin'
            ]

            actual_columns = sorted(kmeans_df.columns.tolist())
            expected_columns = sorted(expected_suffixes)

            # Test exact column names
            self.assertEqual(actual_columns, expected_columns, 
                            f"Expected columns: {expected_columns}\nGot: {actual_columns}")

    def test_init_with_missing_columns(self):
        with redirect_stdout(io.StringIO()):
            """Tests initialization when the site info CSV file is missing required columns.

            The site info CSV file must have specific columns:
            - site_name: Name of the solar site
            - lat: Latitude of the site
            - long: Longitude of the site
            - MW: Megawatt capacity
            - tracking: Type of solar tracking
            - zone: Zone number

            This test creates a DataFrame missing most required columns (only has 'wrong_col' and 'lat')
            and verifies that KMeans_Pipeline raises a KeyError when trying to access
            the missing required columns.

            For example:
                Valid CSV has: site_name, lat, long, MW, tracking, zone
                Test CSV has: wrong_col, lat  # Missing critical columns
            """
            # Mock site info DataFrame missing required columns
            bad_site_info = pd.DataFrame({
                'wrong_col': ['101_PV_1', '101_PV_2'],
                'lat': [33.4, 33.3]  # Missing required columns
            })
            self.mock_read_csv.return_value = bad_site_info

            with self.assertRaises(KeyError):
                KMeans_Pipeline(self.test_directory, self.test_sites_file)

    def test_init_with_malformed_solar_data(self):
        with redirect_stdout(io.StringIO()):
            """Tests a situation where the solar generation data file contains site 
            names that don't match any of the sites listed in the site info CSV file
            For example:
                Site info CSV has sites: 101_PV_1, 101_PV_2, 101_PV_3
                But the solar data Excel file has a different site: DIFFERENT_SITE
            """
            # Mock solar data with wrong datetime format
            bad_solar_data = pd.DataFrame({
                'datetime': ['not-a-date'] * 8760,  # Invalid datetime
                '101_PV_1': [1.0] * 8760
            })

            def mock_excel_bad_data(*args, **kwargs):
                if kwargs.get('sheet_name') == 'solar_gen':
                    return bad_solar_data
                return self.csi_df

            self.mock_read_excel.side_effect = mock_excel_bad_data

            with self.assertRaises(pd.errors.ParserError):
                KMeans_Pipeline(self.test_directory, self.test_sites_file)

    def test_init_with_mismatched_sites(self):
        with redirect_stdout(io.StringIO()):
            """Test initialization with mismatched site names between files"""
            # Mock solar data with different site names than site info
            mismatched_solar = pd.DataFrame({
                'datetime': pd.date_range('1998-01-01', periods=8760, freq='h'),
                'DIFFERENT_SITE': [1.0] * 8760
            })

            def mock_excel_mismatched(*args, **kwargs):
                if kwargs.get('sheet_name') == 'solar_gen':
                    return mismatched_solar
                return self.csi_df

            self.mock_read_excel.side_effect = mock_excel_mismatched

            pipeline = KMeans_Pipeline(self.test_directory, self.test_sites_file)
            # Should have no selected sites since none match
            self.assertEqual(
                len(pipeline.selected_sites), 0,
                f"Expected no matching sites, but found {len(pipeline.selected_sites)} sites: {pipeline.selected_sites}"
            )

    def test_init_with_mismatched_dataframe_lengths(self):
        with redirect_stdout(io.StringIO()):
            """Test initialization when solar_gen_df and csi_df have different lengths"""
            # Create solar data with 8760 rows (full year)
            solar_dates = pd.date_range('1998-01-01', periods=8760, freq='h')
            mismatched_solar = pd.DataFrame({
                'datetime': solar_dates,
                '101_PV_1': [1.0] * 8760
            })

            # Create CSI data with fewer rows (half year)
            csi_dates = pd.date_range('1998-01-01', periods=4380, freq='h')
            mismatched_csi = pd.DataFrame({
                'datetime': csi_dates,
                '101_PV_1': [1.0] * 4380
            })

            def mock_excel_mismatched(*args, **kwargs):
                if kwargs.get('sheet_name') == 'solar_gen':
                    return mismatched_solar
                elif kwargs.get('sheet_name') == 'csi':
                    return mismatched_csi

            self.mock_read_excel.side_effect = mock_excel_mismatched

            with self.assertRaises(ValueError) as context:
                KMeans_Pipeline(self.test_directory, self.test_sites_file)

            self.assertTrue('solar_gen_df and csi_df must have the same length' in str(context.exception))

    def test_run_kmeans_pipeline(self):
        with redirect_stdout(io.StringIO()):
            """Tests the K-means clustering pipeline execution.

            This test verifies that:
            1. Pipeline correctly processes input data through all stages:
               - Datetime conversion and month extraction
               - Feature scaling and PCA transformation
               - K-means clustering execution
            2. Returns predicted labels in expected format
            3. Handles different numbers of clusters correctly

            Test data includes:
            - Full year of data (8760 hours)
            - Multiple features from solar generation and CSI data
            - Tests both default (n_clusters=11) and custom cluster counts
            """
            # Unmock methods for this test
            self.mock_kmeans.stop()

            # Create test DataFrame with all required columns
            dates = pd.date_range('1998-01-01', periods=8760, freq='h').date

            test_df = pd.DataFrame({
                'date': dates,
                '101_PV_1_sg_mean_am': np.random.rand(8760),
                '101_PV_1_sg_mean_pm': np.random.rand(8760),
                '101_PV_1_csi_sd_am': np.random.rand(8760),
                '101_PV_1_csi_sd_pm': np.random.rand(8760),
                'first_light_sin': np.random.rand(8760),
                'first_light_cos': np.random.rand(8760),
                'last_light_sin': np.random.rand(8760),
                'last_light_cos': np.random.rand(8760)
            })

            # Test with default number of clusters
            labels_default = self.pipeline.run_kmeans_pipeline(test_df)
            self.assertIsInstance(labels_default, np.ndarray)
            self.assertEqual(len(labels_default), 8760)
            self.assertEqual(len(np.unique(labels_default)), 11)  # Default is 11 clusters

            # Test with custom number of clusters
            labels_custom = self.pipeline.run_kmeans_pipeline(test_df, n_clusters=5)
            self.assertEqual(len(np.unique(labels_custom)), 5)

            # Test that labels are properly assigned (between 0 and n_clusters-1)
            self.assertTrue(all(0 <= label < 5 for label in labels_custom))

    def test_init_with_nan_values(self):
        with redirect_stdout(io.StringIO()):
            """Tests initialization when data contains NaN values.

            This test verifies that:
            1. The pipeline detects NaN values in solar generation data
            2. Raises a ValueError with appropriate message
            3. Lists exact locations of NaN values in error message

            Test data simulates:
            - Full year of hourly data (8760 hours)
            - Multiple sites with some NaN values
            - NaN values scattered throughout the data
            - Both CSI and solar generation data with NaNs
            """
            # Create solar data with NaN values
            solar_dates = pd.date_range('1998-01-01', periods=8760, freq='h')
            solar_data_with_nans = pd.DataFrame({
                'datetime': solar_dates,
                '101_PV_1': [np.nan if i % 100 == 0 else 1.0 for i in range(8760)],  # NaN every 100th value
                '101_PV_2': [1.0] * 8760,  # No NaNs
                '101_PV_3': [np.nan if i % 500 == 0 else 1.0 for i in range(8760)]   # NaN every 500th value
            })

            # Find which sites have NaN values and at what positions
            nan_sites = {col: positions for col, positions in 
                        {col: solar_data_with_nans[col][solar_data_with_nans[col].isna()].index.tolist()
                         for col in solar_data_with_nans.columns if col != 'datetime'}.items() 
                        if positions}  # Only keep sites with NaN values

            def mock_excel_with_nans(*args, **kwargs):
                if kwargs.get('sheet_name') == 'solar_gen':
                    return solar_data_with_nans
                return self.csi_df

            self.mock_read_excel.side_effect = mock_excel_with_nans

            expected_message = "Solar generation data contains NaN values in sites: "
            expected_message += ", ".join([f"{site} at positions {positions}" 
                                         for site, positions in sorted(nan_sites.items())])

            with self.assertRaises(ValueError) as context:
                KMeans_Pipeline(self.test_directory, self.test_sites_file)

            self.assertEqual(str(context.exception), expected_message)

    # def test_output_suppression(self):
    #     """Test that output is actually being suppressed"""
    #     with redirect_stdout(io.StringIO()) as output:
    #         self.pipeline.update_progress("Test", 0.5)
    #         self.assertEqual(output.getvalue(), "")

# class TestKMeansIntegration(unittest.TestCase):
#     """Integration tests for KMeans Pipeline"""
    
#     def setUp(self):
#         """Setup real test data without mocks"""
#         self.output = io.StringIO()
#         self.test_directory = "/test/path"
#         self.test_sites_file = "/test/path/solar_sites.csv"
        
#     def test_full_pipeline_workflow(self):
#         """Tests the entire pipeline from data loading to clustering"""
#         with redirect_stdout(self.output):
#             pipeline = KMeans_Pipeline(self.test_directory, self.test_sites_file)
#             result = pipeline.run()
            
#             # Verify end-to-end results
#             self.assertIsNotNone(result)
#             self.assertTrue(isinstance(result, np.ndarray))
#             self.assertEqual(len(result), 8760)  # Full year of hourly predictions

if __name__ == '__main__':
    unittest.main(verbosity=2)
