"""
Tests for the occupational code description utility functions.
"""

import unittest

import pandas as pd

from occpast_extensions import (
    load_hisco_descriptions,
    load_descriptions_from_csv,
    load_descriptions_from_dataframe,
    load_descriptions_from_dict,
    get_hisco_description,
    get_description,
    create_code_to_description_mapping,
    add_descriptions_to_dataframe,
    format_input_with_description,
)


class TestLoadHiscoDescriptions(unittest.TestCase):
    def test_returns_dict(self):
        descriptions = load_hisco_descriptions()
        self.assertIsInstance(descriptions, dict)

    def test_contains_string_keys(self):
        descriptions = load_hisco_descriptions()
        # Test that string keys work
        self.assertIn('61110', descriptions)

    def test_contains_int_keys(self):
        descriptions = load_hisco_descriptions()
        # Test that int keys work
        self.assertIn(61110, descriptions)

    def test_special_codes_included_by_default(self):
        descriptions = load_hisco_descriptions()
        # Special codes -1, -2, -3 should be included
        self.assertIn(-1, descriptions)
        self.assertIn(-2, descriptions)
        self.assertIn(-3, descriptions)

    def test_special_codes_can_be_excluded(self):
        descriptions = load_hisco_descriptions(include_special_codes=False)
        # Special codes should not be included
        self.assertNotIn(-1, descriptions)
        self.assertNotIn(-2, descriptions)
        self.assertNotIn(-3, descriptions)


class TestLoadDescriptionsFromDataframe(unittest.TestCase):
    def test_returns_dict(self):
        df = pd.DataFrame({
            'code': ['001', '002', '003'],
            'description': ['Farmer', 'Teacher', 'Doctor']
        })
        descriptions = load_descriptions_from_dataframe(df, 'code', 'description')
        self.assertIsInstance(descriptions, dict)

    def test_correct_mapping(self):
        df = pd.DataFrame({
            'code': ['001', '002', '003'],
            'description': ['Farmer', 'Teacher', 'Doctor']
        })
        descriptions = load_descriptions_from_dataframe(df, 'code', 'description')
        self.assertEqual(descriptions['001'], 'Farmer')
        self.assertEqual(descriptions['002'], 'Teacher')
        self.assertEqual(descriptions['003'], 'Doctor')

    def test_multilingual_mapping(self):
        df = pd.DataFrame({
            'code': ['001', '001', '002', '002'],
            'lang': ['en', 'fr', 'en', 'fr'],
            'description': ['Farmer', 'Agriculteur', 'Teacher', 'Enseignant']
        })
        descriptions = load_descriptions_from_dataframe(df, 'code', 'description', 'lang')
        self.assertEqual(descriptions['001']['en'], 'Farmer')
        self.assertEqual(descriptions['001']['fr'], 'Agriculteur')
        self.assertEqual(descriptions['002']['en'], 'Teacher')
        self.assertEqual(descriptions['002']['fr'], 'Enseignant')


class TestLoadDescriptionsFromDict(unittest.TestCase):
    def test_returns_dict(self):
        mapping = {1: 'Farmer', 2: 'Teacher'}
        descriptions = load_descriptions_from_dict(mapping)
        self.assertIsInstance(descriptions, dict)

    def test_converts_to_strings(self):
        mapping = {1: 'Farmer', 2: 'Teacher', '3': 'Doctor'}
        descriptions = load_descriptions_from_dict(mapping)
        self.assertEqual(descriptions['1'], 'Farmer')
        self.assertEqual(descriptions['2'], 'Teacher')
        self.assertEqual(descriptions['3'], 'Doctor')


class TestGetDescription(unittest.TestCase):
    def test_returns_description(self):
        descriptions = {'001': 'Farmer', '002': 'Teacher'}
        desc = get_description('001', descriptions)
        self.assertEqual(desc, 'Farmer')

    def test_returns_default_for_missing(self):
        descriptions = {'001': 'Farmer', '002': 'Teacher'}
        desc = get_description('999', descriptions, default='Unknown')
        self.assertEqual(desc, 'Unknown')

    def test_handles_int_key(self):
        descriptions = {'001': 'Farmer', '002': 'Teacher'}
        desc = get_description(1, descriptions, default='Unknown')
        # Should convert 1 to '1' and look up
        self.assertEqual(desc, 'Unknown')  # '1' != '001'


class TestGetHiscoDescription(unittest.TestCase):
    def setUp(self):
        self.descriptions = load_hisco_descriptions()

    def test_string_key_lookup(self):
        desc = get_hisco_description('61110', self.descriptions)
        self.assertIsInstance(desc, str)
        self.assertNotEqual(desc, 'Unknown occupation')

    def test_int_key_lookup(self):
        desc = get_hisco_description(61110, self.descriptions)
        self.assertIsInstance(desc, str)
        self.assertNotEqual(desc, 'Unknown occupation')

    def test_default_value_for_missing_code(self):
        desc = get_hisco_description(99999999, self.descriptions)
        self.assertEqual(desc, 'Unknown occupation')

    def test_custom_default_value(self):
        desc = get_hisco_description(99999999, self.descriptions, default='N/A')
        self.assertEqual(desc, 'N/A')

    def test_loads_descriptions_if_not_provided(self):
        desc = get_hisco_description('61110')
        self.assertIsInstance(desc, str)
        self.assertNotEqual(desc, 'Unknown occupation')


class TestCreateCodeToDescriptionMapping(unittest.TestCase):
    def test_returns_dict(self):
        mapping = create_code_to_description_mapping()
        self.assertIsInstance(mapping, dict)

    def test_contains_integer_keys(self):
        mapping = create_code_to_description_mapping()
        # Keys should be integers (the model's internal code)
        for key in list(mapping.keys())[:10]:
            self.assertIsInstance(key, int)

    def test_code_0_maps_to_description(self):
        mapping = create_code_to_description_mapping()
        # Code 0 should be "Non work related title"
        self.assertIn(0, mapping)
        self.assertIsInstance(mapping[0], str)


class TestAddDescriptionsToDataframe(unittest.TestCase):
    def test_adds_description_column(self):
        df = pd.DataFrame({'hisco_1': ['61110', '95120', '-1']})
        result = add_descriptions_to_dataframe(df)
        self.assertIn('hisco_description', result.columns)

    def test_custom_column_names(self):
        df = pd.DataFrame({'my_hisco': ['61110', '95120']})
        result = add_descriptions_to_dataframe(
            df,
            hisco_column='my_hisco',
            description_column='my_desc'
        )
        self.assertIn('my_desc', result.columns)

    def test_does_not_modify_original(self):
        df = pd.DataFrame({'hisco_1': ['61110']})
        original_columns = list(df.columns)
        _ = add_descriptions_to_dataframe(df)
        self.assertEqual(list(df.columns), original_columns)


class TestFormatInputWithDescription(unittest.TestCase):
    def setUp(self):
        self.descriptions = load_hisco_descriptions()

    def test_default_format(self):
        result = format_input_with_description(
            'farmer',
            '61110',
            self.descriptions
        )
        self.assertIn('farmer', result)
        self.assertIn('[DESC:', result)

    def test_custom_format(self):
        result = format_input_with_description(
            'carpenter',
            '95410',
            self.descriptions,
            format_template='{occ} ({desc})'
        )
        self.assertIn('carpenter', result)
        self.assertIn('(', result)
        self.assertIn(')', result)

    def test_loads_descriptions_if_not_provided(self):
        result = format_input_with_description('farmer', '61110')
        self.assertIn('farmer', result)
        self.assertIn('[DESC:', result)


if __name__ == '__main__':
    unittest.main()
