import matplotlib.pyplot as plt
from pandas.plotting import table


def plot_multiple_waveforms(waveforms, sample_rate=10000, labels=None, title='Waveform Comparison'):
    """
    Plots multiple waveforms on the same chart.

    :param waveforms: List of waveforms to plot.
    :param sample_rate: The sample rate of the waveforms. Default is 10000.
    :param labels: List of labels for the waveforms. If None, default labels are used.
    :param title: The title of the plot.
    """
    assert len(waveforms) > 0, "No waveforms provided for plotting"

    # Use default labels if none are provided
    if labels is None:
        labels = [f'Waveform {i+1}' for i in range(len(waveforms))]

    assert len(waveforms) == len(labels), "Number of waveforms and labels must match"

    # Generate a time axis
    time_axis = [i / sample_rate for i in range(len(waveforms[0]))]

    # Plot each waveform
    for i, wave in enumerate(waveforms):
        plt.plot(time_axis, wave, label=labels[i])

    # Adding labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()


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
    ax.axis('off')

    # Create the table in the plot with manually adjusted column widths
    the_table = table(ax, df_formatted, loc='center', cellLoc='center', colWidths=col_widths)

    # Set font size for all cells in the table
    for _, cell in the_table.get_celld().items():
        cell.set_text_props(fontsize=font_size)

        cell.set_edgecolor('lightgrey')  # Optionally adjusts cell border color

    # Save the figure to a file
    plt.savefig(str(filename), bbox_inches='tight', dpi=500)
    plt.close('all')
    print(f'DataFrame exported as image to {filename}')
