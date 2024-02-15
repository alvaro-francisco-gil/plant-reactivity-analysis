from PlantReactivityAnalysis.features.features_dataset import FeaturesDataset
from sklearn.decomposition import PCA

norm_letters_signal_dataset_path = r"../data/processed/feat_norm_letters_1_1_dataset.pkl"
raw_letters_signal_dataset_path = r"../data/processed/feat_raw_letters_1_1_dataset.pkl"

RQS = [1, 2]


def collect_all_rqs_data(corr_threshold=0, pca_dim=42):
    # Function mappings
    dataset_functions = [
        return_ct_datasets1,
        return_ct_datasets2,
        return_ct_datasets3,
        return_ct_datasets4,
        return_ct_datasets5,
        return_ct_datasets7,
    ]

    pca_functions = [
        return_pca_datasets_9,
        return_pca_datasets_10,
        return_pca_datasets_11,
        return_pca_datasets_12,
        return_pca_datasets_13,
        return_pca_datasets_15
    ]

    all_data = {}

    # Iterate through each dataset function for correlation threshold
    for dataset_func in dataset_functions:
        func_name = dataset_func.__name__
        # Assuming the identifier is at the end of the function name, following the last underscore
        identifier = int(func_name[-1:])
        print(f"\nPROCESSING DATASET {identifier}")
        rqs_data = dataset_func(corr_threshold)
        all_data[identifier] = rqs_data

    # Iterate through each dataset function for PCA
    for pca_func in pca_functions:
        func_name = pca_func.__name__
        # Extracting identifier
        identifier = int(func_name.split('_')[-1])
        print(f"\nPROCESSING DATASET {identifier}")
        rqs_data = pca_func(pca_dim)
        all_data[identifier] = rqs_data

    return all_data


def return_ct_datasets1(corr_threshold=0):
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_ct_datasets2(corr_threshold=0):
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_ct_datasets3(corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_ct_datasets4(corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_ct_datasets5(corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_ct_datasets6(corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_ct_datasets7(corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_ct_datasets8(corr_threshold=0):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_pca_datasets_9(n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_pca_datasets_10(n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_pca_datasets_11(n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_pca_datasets_12(n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_pca_datasets_13(n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    # Assuming FeaturesDataset and other necessary utilities are defined elsewhere
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_pca_datasets_14(n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    # Assuming FeaturesDataset and other necessary utilities are defined elsewhere
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_pca_datasets_15(n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    # Assuming FeaturesDataset and other necessary utilities are defined elsewhere
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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


def return_pca_datasets_16(n_components=None):
    """
    Load datasets, split into training and test sets, then apply PCA to reduce dimensionality
    based on the specified number of components.

    Parameters:
    - n_components: Number of principal components to keep. If None, PCA is not applied.

    Returns:
    A dictionary of DataFrames for each research question, with each DataFrame being the result of PCA.
    """

    # Assuming FeaturesDataset and other necessary utilities are defined elsewhere
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in RQS:
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
