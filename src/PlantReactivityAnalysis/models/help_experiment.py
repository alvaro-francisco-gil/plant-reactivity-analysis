from PlantReactivityAnalysis.features.features_dataset import FeaturesDataset
norm_letters_signal_dataset_path = r"../data/processed/feat_norm_letters_1_1_dataset.pkl"
raw_letters_signal_dataset_path = r"../data/processed/feat_raw_letters_1_1_dataset.pkl"


def collect_all_rqs_data(reduce_variables=True):
    # Function mappings
    dataset_functions = [
        return_rqs_dataset1,
        return_rqs_dataset2,
        return_rqs_dataset3,
        return_rqs_dataset4,
        return_rqs_dataset5,
        return_rqs_dataset6,
        return_rqs_dataset7,
        return_rqs_dataset8,
    ]

    all_data = {}

    # Iterate through each dataset function
    for i, dataset_func in enumerate(dataset_functions, start=1):
        print("Processing Dataset", i)
        # Call the dataset function
        rqs_data = dataset_func(reduce_variables)
        # Store its return value in the all_data dictionary
        all_data[i] = rqs_data

    return all_data


def return_rqs_dataset1(reduce_variables=True):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in [1, 2, 5]:
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        if reduce_variables:
            # Reduce the features that are correlated in the training data
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=0.8)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(feat_stats.head(10))

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_rqs_dataset2(reduce_variables=True):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in [1, 2, 5]:
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        if reduce_variables:
            # Reduce the features that are correlated in the training data
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=0.8)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(feat_stats.head(10))

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_rqs_dataset3(reduce_variables=True):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in [1, 2, 5]:
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        if reduce_variables:
            # Reduce the features that are correlated in the training data
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=0.8)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(feat_stats.head(10))

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_rqs_dataset4(reduce_variables=True):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=norm_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in [1, 2, 5]:
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        if reduce_variables:
            # Reduce the features that are correlated in the training data
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=0.8)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(feat_stats.head(10))

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_rqs_dataset5(reduce_variables=True):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in [1, 2, 5]:
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        if reduce_variables:
            # Reduce the features that are correlated in the training data
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=0.8)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(feat_stats.head(10))

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_rqs_dataset6(reduce_variables=True):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=False, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in [1, 2, 5]:
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        if reduce_variables:
            # Reduce the features that are correlated in the training data
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=0.8)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(feat_stats.head(10))

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_rqs_dataset7(reduce_variables=True):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in [1, 2, 5]:
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        if reduce_variables:
            # Reduce the features that are correlated in the training data
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=0.8)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(feat_stats.head(10))

        # Normalize Features
        normalization_params = train_norm_dataset.normalize_features()
        test_norm_dataset.apply_normalization(normalization_params)

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs


def return_rqs_dataset8(reduce_variables=True):

    # Dataset 1
    norm_dataset = FeaturesDataset.load(file_path=raw_letters_signal_dataset_path)
    norm_dataset.prepare_dataset(drop_constant=True, drop_flatness_columns=True, drop_nan_columns=True)
    rqs = {}
    for x in [1, 2, 5]:
        rq = norm_dataset.return_subset_given_research_question(x)
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(split_by_wav=False, test_size=0.2,
                                                                    val_size=0, random_state=42)

        if reduce_variables:
            # Reduce the features that are correlated in the training data
            train_cols, feat_stats = train_norm_dataset.reduce_features_based_on_target(corr_threshold=0.8)
            test_norm_dataset.keep_only_specified_variable_columns(train_cols)
            print(feat_stats.head(10))

        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features

        rqs[x] = train_df, test_df

    return rqs
