from PlantReactivityAnalysis.features.features_dataset import FeaturesDataset
from sklearn.decomposition import PCA


def get_dataset_by_question(path, rqs, corr_threshold=0.8, ttest=False):

    dataset = FeaturesDataset.load(file_path=path)
    dataset.make_final_dataset()
    questions_data = {}

    for x in rqs:
        print("\n# Research Question: ", x)
        rq = dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)
        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        train_cols, _ = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold,
                                                                           print_test=True)
        test_norm_dataset.keep_only_specified_variable_columns(train_cols)
        print(f"Reduced features based on correlation threshold of {corr_threshold}")

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features
        questions_data[x] = train_df, test_df

    return questions_data


def get_datasets_by_ids(path, rqs, corr_threshold=0, pca_dim=42,
                        dataset_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]):

    all_data = {}

    for ds_index in dataset_ids:
        print(f"\nPROCESSING DATASET {ds_index}")

        questions_data = {}
        dataset = FeaturesDataset.load(file_path=path)

        rows_to_drop = []
        if ds_index in [3, 4, 11, 12]:
            rows_to_drop = dataset.objective_features[(dataset.objective_features['flatness_ratio_1000'] > 0.75) &
                                                      (dataset.objective_features['flatness_ratio_500'] > 0.85) &
                                                      (dataset.objective_features['flatness_ratio_100'] > 0.999)]\
                                                        .index.to_list()
        elif ds_index in [5, 6, 13, 14]:
            rows_to_drop = dataset.objective_features[(dataset.objective_features['flatness_ratio_1000'] > 0.75) |
                                                      (dataset.objective_features['flatness_ratio_500'] > 0.85) |
                                                      (dataset.objective_features['flatness_ratio_100'] > 0.999)]\
                                                        .index.to_list()
        elif ds_index in [7, 8, 15, 16]:
            rows_to_drop = dataset.objective_features[(dataset.objective_features['flatness_ratio_1000'] > 0.6) |
                                                      (dataset.objective_features['flatness_ratio_500'] > 0.85) |
                                                      (dataset.objective_features['flatness_ratio_100'] > 0.999)]\
                                                        .index.to_list()

        dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
        dataset.drop_rows(rows_to_drop)

        for x in rqs:
            print("\n# Research Question: ", x)
            rq = dataset.return_subset_given_research_question(x)
            train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                        val_size=0, random_state=42)
            print('-Train distribution-')
            train_norm_dataset.print_target_distribution()
            print('-Test distribution-')
            test_norm_dataset.print_target_distribution()

            if ds_index % 2 == 1:
                # Normalize Features
                normalization_params = train_norm_dataset.normalize_features()
                test_norm_dataset.apply_normalization(normalization_params)

            if ds_index <= 8:

                train_cols, _ = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold,
                                                                                   print_ttest=True)
                test_norm_dataset.keep_only_specified_variable_columns(train_cols)
                print(f"Reduced features based on correlation threshold of {corr_threshold}")

                train_df = train_norm_dataset.objective_features
                test_df = test_norm_dataset.objective_features
                questions_data[x] = train_df, test_df

            else:
                # Get train and test targets
                train_targets = train_norm_dataset.features[train_norm_dataset.target_column_name]
                train_norm_dataset.drop_columns([train_norm_dataset.target_column_name])
                test_targets = test_norm_dataset.features[test_norm_dataset.target_column_name]
                test_norm_dataset.drop_columns([test_norm_dataset.target_column_name])

                # Fit PCA on training data and transform both training and test data
                pca = PCA(n_components=pca_dim)
                train_df = pca.fit_transform(train_norm_dataset.objective_features)
                test_df = pca.transform(test_norm_dataset.objective_features)

                questions_data[x] = train_df, train_targets, test_df, test_targets

        all_data[ds_index] = questions_data

    return all_data
