from PlantReactivityAnalysis.features.features_dataset import FeaturesDataset
from sklearn.decomposition import PCA


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


def collect_all_rqs_data(norm_path, raw_path, rqs, dataset_ids=[1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 15],
                         corr_threshold=0, pca_dim=42):
    """
    Collects data from specified datasets using either correlation threshold or PCA based on the dataset function.

    :param norm_path: Path to normalized datasets.
    :param raw_path: Path to raw datasets.
    :param rqs: Research questions to process.
    :param dataset_ids: List of integers specifying which dataset functions to run.
    :param corr_threshold: Correlation threshold parameter.
    :param pca_dim: PCA dimension parameter.
    :return: A dictionary with processed data for each requested dataset.
    """
    # Function mappings with associated dataset identifiers
    dataset_function_map = {
        1: return_ct_datasets1,
        2: return_ct_datasets2,
        3: return_ct_datasets3,
        4: return_ct_datasets4,
        5: return_ct_datasets5,
        7: return_ct_datasets7,
        9: return_pca_datasets_9,
        10: return_pca_datasets_10,
        11: return_pca_datasets_11,
        12: return_pca_datasets_12,
        13: return_pca_datasets_13,
        15: return_pca_datasets_15,
    }

    all_data = {}

    # Iterate over requested dataset identifiers
    for identifier in dataset_ids:
        print(f"\nPROCESSING DATASET {identifier}")

        # Retrieve the function based on identifier
        dataset_func = dataset_function_map.get(identifier)
        if not dataset_func:
            print(f"Dataset function for identifier {identifier} not found.")
            continue

        # Decide which path to use based on dataset identifier
        path = norm_path if identifier in [1, 2, 3, 4, 9, 10, 11, 12] else raw_path

        # Decide parameter based on function type (corr_threshold or pca_dim)
        parameter = corr_threshold if identifier <= 7 else pca_dim

        # Execute the function
        rqs_data = dataset_func(path, rqs, parameter)
        all_data[identifier] = rqs_data

    return all_data


def return_ct_datasets1(path, rq_list, corr_threshold=0):
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)
        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Reduce features based on correlation threshold if specified
        if corr_threshold > 0:
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(f"Reduced features based on correlation threshold of {corr_threshold}")

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_ct_datasets2(path, rq_list, corr_threshold=0):
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)
        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Reduce features based on correlation threshold if specified
        if corr_threshold > 0:
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(f"Reduced features based on correlation threshold of {corr_threshold}")

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_ct_datasets3(path, rq_list, corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Reduce features based on correlation threshold if specified
        if corr_threshold > 0:
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(f"Reduced features based on correlation threshold of {corr_threshold}")

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_ct_datasets4(path, rq_list, corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Reduce features based on correlation threshold if specified
        if corr_threshold > 0:
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(f"Reduced features based on correlation threshold of {corr_threshold}")

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_ct_datasets5(path, rq_list, corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Reduce features based on correlation threshold if specified
        if corr_threshold > 0:
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(f"Reduced features based on correlation threshold of {corr_threshold}")

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_ct_datasets6(path, rq_list, corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Reduce features based on correlation threshold if specified
        if corr_threshold > 0:
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(f"Reduced features based on correlation threshold of {corr_threshold}")

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_ct_datasets7(path, rq_list, corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Reduce features based on correlation threshold if specified
        if corr_threshold > 0:
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(f"Reduced features based on correlation threshold of {corr_threshold}")

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_ct_datasets8(path, rq_list, corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Reduce features based on correlation threshold if specified
        if corr_threshold > 0:
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=corr_threshold)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(f"Reduced features based on correlation threshold of {corr_threshold}")

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_pca_datasets_9(path, rq_list, n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        # Get train targets and remove column
        train_targets = train_norm_dataset.features[train_norm_dataset.target_column_name]
        train_norm_dataset.drop_columns([train_norm_dataset.target_column_name])

        # Get train targets and remove column
        test_targets = test_norm_dataset.features[test_norm_dataset.target_column_name]
        test_norm_dataset.drop_columns([test_norm_dataset.target_column_name])

        # Fit PCA on training data and transform both training and test data
        pca = PCA(n_components=n_components)
        train_df = pca.fit_transform(train_norm_dataset.objective_features)
        test_df = pca.transform(test_norm_dataset.objective_features)

        rqs[x] = train_df, train_targets, test_df, test_targets

    return rqs


def return_pca_datasets_10(path, rq_list, n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Get train targets and remove column
        train_targets = train_norm_dataset.features[train_norm_dataset.target_column_name]
        train_norm_dataset.drop_columns([train_norm_dataset.target_column_name])

        # Get train targets and remove column
        test_targets = test_norm_dataset.features[test_norm_dataset.target_column_name]
        test_norm_dataset.drop_columns([test_norm_dataset.target_column_name])

        # Fit PCA on training data and transform both training and test data
        pca = PCA(n_components=n_components)
        train_df = pca.fit_transform(train_norm_dataset.objective_features)
        test_df = pca.transform(test_norm_dataset.objective_features)

        rqs[x] = train_df, train_targets, test_df, test_targets

    return rqs


def return_pca_datasets_11(path, rq_list, n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        # Get train targets and remove column
        train_targets = train_norm_dataset.features[train_norm_dataset.target_column_name]
        train_norm_dataset.drop_columns([train_norm_dataset.target_column_name])

        # Get train targets and remove column
        test_targets = test_norm_dataset.features[test_norm_dataset.target_column_name]
        test_norm_dataset.drop_columns([test_norm_dataset.target_column_name])

        # Fit PCA on training data and transform both training and test data
        pca = PCA(n_components=n_components)
        train_df = pca.fit_transform(train_norm_dataset.objective_features)
        test_df = pca.transform(test_norm_dataset.objective_features)

        rqs[x] = train_df, train_targets, test_df, test_targets

    return rqs


def return_pca_datasets_12(path, rq_list, n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Get train targets and remove column
        train_targets = train_norm_dataset.features[train_norm_dataset.target_column_name]
        train_norm_dataset.drop_columns([train_norm_dataset.target_column_name])

        # Get train targets and remove column
        test_targets = test_norm_dataset.features[test_norm_dataset.target_column_name]
        test_norm_dataset.drop_columns([test_norm_dataset.target_column_name])

        # Fit PCA on training data and transform both training and test data
        pca = PCA(n_components=n_components)
        train_df = pca.fit_transform(train_norm_dataset.objective_features)
        test_df = pca.transform(test_norm_dataset.objective_features)

        rqs[x] = train_df, train_targets, test_df, test_targets

    return rqs


def return_pca_datasets_13(path, rq_list, n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    # Assuming FeaturesDataset and other necessary utilities are defined elsewhere
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        # Get train targets and remove column
        train_targets = train_norm_dataset.features[train_norm_dataset.target_column_name]
        train_norm_dataset.drop_columns([train_norm_dataset.target_column_name])

        # Get train targets and remove column
        test_targets = test_norm_dataset.features[test_norm_dataset.target_column_name]
        test_norm_dataset.drop_columns([test_norm_dataset.target_column_name])

        # Fit PCA on training data and transform both training and test data
        pca = PCA(n_components=n_components)
        train_df = pca.fit_transform(train_norm_dataset.objective_features)
        test_df = pca.transform(test_norm_dataset.objective_features)

        rqs[x] = train_df, train_targets, test_df, test_targets

    return rqs


def return_pca_datasets_14(path, rq_list, n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    # Assuming FeaturesDataset and other necessary utilities are defined elsewhere
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Get train targets and remove column
        train_targets = train_norm_dataset.features[train_norm_dataset.target_column_name]
        train_norm_dataset.drop_columns([train_norm_dataset.target_column_name])

        # Get train targets and remove column
        test_targets = test_norm_dataset.features[test_norm_dataset.target_column_name]
        test_norm_dataset.drop_columns([test_norm_dataset.target_column_name])

        # Fit PCA on training data and transform both training and test data
        pca = PCA(n_components=n_components)
        train_df = pca.fit_transform(train_norm_dataset.objective_features)
        test_df = pca.transform(test_norm_dataset.objective_features)

        rqs[x] = train_df, train_targets, test_df, test_targets

    return rqs


def return_pca_datasets_15(path, rq_list, n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    # Assuming FeaturesDataset and other necessary utilities are defined elsewhere
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        # Get train targets and remove column
        train_targets = train_norm_dataset.features[train_norm_dataset.target_column_name]
        train_norm_dataset.drop_columns([train_norm_dataset.target_column_name])

        # Get train targets and remove column
        test_targets = test_norm_dataset.features[test_norm_dataset.target_column_name]
        test_norm_dataset.drop_columns([test_norm_dataset.target_column_name])

        # Fit PCA on training data and transform both training and test data
        pca = PCA(n_components=n_components)
        train_df = pca.fit_transform(train_norm_dataset.objective_features)
        test_df = pca.transform(test_norm_dataset.objective_features)

        rqs[x] = train_df, train_targets, test_df, test_targets

    return rqs


def return_pca_datasets_16(path, rq_list, n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    # Assuming FeaturesDataset and other necessary utilities are defined elsewhere
    norm_dataset = FeaturesDataset.load(file_path=path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in rq_list:
        print("\n# Research Question: ", x)
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        print('-Train distribution-')
        train_norm_dataset.print_target_distribution()
        print('-Test distribution-')
        test_norm_dataset.print_target_distribution()

        # Get train targets and remove column
        train_targets = train_norm_dataset.features[train_norm_dataset.target_column_name]
        train_norm_dataset.drop_columns([train_norm_dataset.target_column_name])

        # Get train targets and remove column
        test_targets = test_norm_dataset.features[test_norm_dataset.target_column_name]
        test_norm_dataset.drop_columns([test_norm_dataset.target_column_name])

        # Fit PCA on training data and transform both training and test data
        pca = PCA(n_components=n_components)
        train_df = pca.fit_transform(train_norm_dataset.objective_features)
        test_df = pca.transform(test_norm_dataset.objective_features)

        rqs[x] = train_df, train_targets, test_df, test_targets

    return rqs
