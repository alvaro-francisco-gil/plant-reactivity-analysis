from PlantReactivityAnalysis.features.features_dataset import FeaturesDataset


def get_dataset_by_question(path, rqs, corr_threshold=0.8, ttest=False):
    """
    Loads a dataset and prepares subsets based on specified research questions (RQs),
    applying feature reduction based on correlation threshold and optionally a t-test.

    :param path: Path to the dataset file.
    :param rqs: A list of research questions, used to create subsets of the dataset.
    :param corr_threshold: Correlation threshold for feature reduction. Features with a correlation
                           to the target variable above this threshold will be kept.
    :param ttest: Boolean indicating whether to perform a t-test for feature selection. Currently,
                  this parameter is not used in the function.
    :return: A dictionary where keys are RQs and values are tuples containing the train and test DataFrames
             after feature reduction and dataset splitting.
    """
    # Load the dataset from the specified path and prepare it for analysis
    dataset = FeaturesDataset.load(file_path=path)
    dataset.make_final_dataset()
    questions_data = {}

    # Process each research question
    for x in rqs:
        print("\n# Research Question: ", x)

        # Get a subset of the dataset relevant to the current research question
        rq = dataset.return_subset_given_research_question(x)

        # Split the subset into training and testing datasets
        train_norm_dataset, _, test_norm_dataset = rq.split_dataset(
            split_by_wav=False, test_size=0.2, val_size=0, random_state=42
        )

        # Display the target distribution in the training and testing datasets
        print("-Train distribution-")
        train_norm_dataset.print_target_distribution()
        print("-Test distribution-")
        test_norm_dataset.print_target_distribution()

        # Reduce features in the training dataset based on correlation with the target
        train_cols, _ = train_norm_dataset.reduce_features_based_on_target(
            corr_threshold=corr_threshold, print_test=True
        )
        # Reduce features in the testing dataset to match the training dataset's features
        test_norm_dataset.keep_only_specified_variable_columns(train_cols)
        print(f"Reduced features based on correlation threshold of {corr_threshold}")

        # Extract the final features for training and testing
        train_df = train_norm_dataset.objective_features
        test_df = test_norm_dataset.objective_features
        # Store the training and testing DataFrames for the current research question
        questions_data[x] = train_df, test_df

    return questions_data
