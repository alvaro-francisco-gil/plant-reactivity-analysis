import matplotlib.pyplot as plt


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
