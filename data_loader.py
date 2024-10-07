import pandas as pd

class DataLoader:
    def __init__(self, demographic_columns):
        self.demographic_columns = demographic_columns
        self.features_mdd = {
            't1_intensity': {},
            'thickness': {},
            'sulcal_depth': {},
            'surface_area': {},
            'volume': {}
        }

    def load_and_filter_groups(self, merged_groups_dict_t1w, merged_groups_dict_str):
        """Load and filter data for SI/SA with MDD and SI/SA without MDD groups."""
        filtered_merged_groups_dict_t1w = self._filter_groups(merged_groups_dict_t1w)
        filtered_merged_groups_dict_str = self._filter_groups(merged_groups_dict_str)
        return filtered_merged_groups_dict_t1w, filtered_merged_groups_dict_str

    def _filter_groups(self, merged_groups_dict):
        """Helper function to filter 'si_sa_with_mdd_pass' and 'si_sa_no_mdd_pass' groups."""
        filtered_dict = {}
        for dataset_name, groups in merged_groups_dict.items():
            group_data = []
            for group_name in ['si_sa_with_mdd_pass', 'si_sa_no_mdd_pass']:
                if group_name in groups:
                    data = groups[group_name].copy()
                    data['group'] = group_name  # Add group membership column
                    group_data.append(data)
            if group_data:
                filtered_dict[dataset_name] = pd.concat(group_data)
        return filtered_dict

    def merge_features(self, filtered_merged_groups_dict_t1w, filtered_merged_groups_dict_str):
        """Merge T1 intensity, cortical thickness, sulcal depth, surface area, and volume features."""
        self._merge_by_parcellation(filtered_merged_groups_dict_t1w, 't1_intensity')
        self._merge_by_parcellation(filtered_merged_groups_dict_str, 'thickness')
        self._merge_by_parcellation(filtered_merged_groups_dict_str, 'sulcal_depth')
        self._merge_by_parcellation(filtered_merged_groups_dict_str, 'surface_area')
        self._merge_by_parcellation(filtered_merged_groups_dict_str, 'volume')

    def _merge_by_parcellation(self, filtered_data, feature_name):
        """Helper function to merge data for each parcellation type and feature."""
        for dataset_name, group_df in filtered_data.items():
            if 'src_subject_id' not in group_df.columns:
                print(f"'src_subject_id' is missing from {dataset_name}. Skipping this dataset.")
                continue

            si_sa_with_mdd_data = group_df[group_df['group'] == 'si_sa_with_mdd_pass'].copy()
            si_sa_no_mdd_data = group_df[group_df['group'] == 'si_sa_no_mdd_pass'].copy()

            if si_sa_with_mdd_data.empty or si_sa_no_mdd_data.empty:
                print(f"No data available for one of the groups in {dataset_name}. Skipping this dataset.")
                continue

            if 'dsk' in dataset_name:
                self.features_mdd[feature_name]['desikan_with_mdd'] = si_sa_with_mdd_data
                self.features_mdd[feature_name]['desikan_no_mdd'] = si_sa_no_mdd_data
            elif 'dst' in dataset_name:
                self.features_mdd[feature_name]['destrieux_with_mdd'] = si_sa_with_mdd_data
                self.features_mdd[feature_name]['destrieux_no_mdd'] = si_sa_no_mdd_data
            elif 'fzy' in dataset_name:
                self.features_mdd[feature_name]['fuzzy_with_mdd'] = si_sa_with_mdd_data
                self.features_mdd[feature_name]['fuzzy_no_mdd'] = si_sa_no_mdd_data

    def get_features_mdd(self):
        """Return the features dictionary containing merged data for MDD and non-MDD groups."""
        return self.features_mdd

# Usage example:
# demographic_columns = ['demo_gender_id_v2', 'demo_brthdat_v2', 'demo_prnt_marital_v2', 'race', 'highest_household_education', 'household_income_per_year']
# data_loader = DataLoader(demographic_columns)
# filtered_merged_groups_t1w, filtered_merged_groups_str = data_loader.load_and_filter_groups(merged_groups_dict_t1w, merged_groups_dict_str)
# data_loader.merge_features(filtered_merged_groups_t1w, filtered_merged_groups_str)
# features_mdd = data_loader.get_features_mdd()

