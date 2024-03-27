import matplotlib.pyplot as plt
from pandas.plotting import table
import os
import seaborn as sns

from PlantReactivityAnalysis.config import FIGURES_DIR


def plot_multiple_waveforms(waveforms, sample_rate=10000, labels=None, title="Waveform Comparison",
                            save_path=None, show_legend=True, figsize=(10, 6)):
    """
    Plots multiple waveforms on the same chart. Optionally saves the figure to a specified path.

    :param waveforms: List of waveforms to plot.
    :param sample_rate: The sample rate of the waveforms. Default is 10000.
    :param labels: List of labels for the waveforms. If None, default labels are used.
    :param title: The title of the plot.
    :param save_path: Full path  where the figure should be saved. If None, the figure is not saved.
    :param show_legend: Boolean indicating whether to display the legend. Default is True.
    :param figsize: Tuple indicating the size of the figure (width, height) in inches. Default is (10, 6).
    """
    assert len(waveforms) > 0, "No waveforms provided for plotting"

    if labels is None:
        labels = [f"Waveform {i+1}" for i in range(len(waveforms))]

    assert len(waveforms) == len(labels), "Number of waveforms and labels must match"

    plt.figure(figsize=figsize)
    time_axis = [i / sample_rate for i in range(len(waveforms[0]))]

    for i, wave in enumerate(waveforms):
        plt.plot(time_axis, wave, label=labels[i] if show_legend else "_nolegend_")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title(title)

    if show_legend:
        plt.legend()

    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()

    plt.clf()


def plot_signals_with_info(ids, signals, df, columns, output_folder, sampling_rate=10000, title_font_size=12.5):
    """
    Plots each signal with specified column values displayed on top, assuming the x-axis represents time in seconds,
    saves the plots as images in the specified folder, and allows setting the title font size.

    Parameters:
    - ids: List of integer ids indicating which signals to plot.
    - signals: Array of signals where each signal corresponds to a row in the df.
    - df: DataFrame containing the features for each signal.
    - columns: List of column names in the df whose values are to be displayed on top of each plot.
    - sampling_rate: The sampling rate of the signals (default is 1 sample per second).
    - output_folder: Path to the folder where the images will be saved.
    - title_font_size: Font size for the plot titles.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for id in ids:
        # Ensure id is within the range of signals and dataframe
        if id < len(signals) and id < len(df):
            # Extract the signal
            signal = signals[id]
            # Generate time vector for x-axis, assuming time starts at 0
            time = [i / sampling_rate for i in range(len(signal))]
            # Extract the values of the specified columns for the current id
            info = df.loc[id, columns].to_dict()
            # Format float values to 3 decimal places
            info_str = ", ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in info.items())

            # Plotting the signal
            plt.figure(figsize=(10, 4))
            plt.plot(time, signal)
            plt.title(f"Signal ID: {id} | {info_str}", fontsize=title_font_size)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')

            # Save the plot as an image file
            image_filename = f"Signal_{id}.png"
            plt.savefig(os.path.join(output_folder, image_filename))
            plt.show()
            plt.close()  # Close the figure to free memory
        else:
            print(f"ID {id} is out of range.")


def format_number(x):
    """Format number with two decimals in standard or scientific notation."""
    if isinstance(x, (int, float)):
        return f"{x:.2f}" if abs(x) >= 0.01 else f"{x:.2e}"
    return x


def export_df_to_image_formatted(df, filename, figsize=(20, 10), col_widths=None, font_size=10):
    """
    Exports a pandas DataFrame to an image file with manual column widths and specified font size.

    Parameters:
    - df: The pandas DataFrame to export.
    - filename: The filename for the saved image.
    - figsize: Tuple representing the figure size.
    - col_widths: List of widths for each column. If None, defaults will be used.
    - font_size: Font size for the text in the table.
    """
    # Apply formatting to each value in the DataFrame
    df_formatted = df.apply(lambda x: x.apply(format_number) if x.dtype == float else x)

    # Create a figure and a subplot without axes for the table
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Create the table in the plot with manually adjusted column widths
    the_table = table(ax, df_formatted, loc="center", cellLoc="center", colWidths=col_widths)

    # Set font size for all cells in the table
    for _, cell in the_table.get_celld().items():
        cell.set_text_props(fontsize=font_size)

        cell.set_edgecolor("lightgrey")  # Optionally adjusts cell border color

    # Save the figure to a file
    plt.savefig(str(filename), bbox_inches="tight", dpi=500)
    plt.close("all")
    print(f"DataFrame exported as image to {filename}")


def plot_confusion_matrix(conf_matrix, title, xticklabels=['Predicted Negative', 'Predicted Positive'],
                          yticklabels=['Actual Negative', 'Actual Positive']):
    """
    Plots a confusion matrix as a heatmap.

    Parameters:
    - conf_matrix: np.array, the confusion matrix to be plotted.
    - title: str, the title of the plot.
    - xticklabels: list of str, labels for the x-axis. Defaults to ['Predicted Negative', 'Predicted Positive'].
    - yticklabels: list of str, labels for the y-axis. Defaults to ['Actual Negative', 'Actual Positive'].
    """
    # Set the context and a larger font for clarity
    sns.set(context='talk', style='white')

    # Create the heatmap for the confusion matrix
    plt.figure(figsize=(5, 5))  # Set the figure size
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=xticklabels, yticklabels=yticklabels)

    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()  # Adjust the layout to make room for the labels
    file_path = os.path.join(FIGURES_DIR, title)
    plt.savefig(file_path)
    plt.show()
