from asyncio import sleep

import serial

s = serial.Serial('/dev/tty.usbmodem1101')
c = 0
i = 0

max_numpix = 64 * 32
numpix = max_numpix

num_samples = 1000

times = []

def send(st):
    print(st)
    s.write(bytes(st, 'utf-8'))
    print(s.readline())

for _ in range(num_samples):
    msg = ''
    for x in range(numpix):
        msg += str(14)
    send(msg)
    sleep(0.04)

import matplotlib.pyplot as plt
from typing import List, Optional


def plot_histogram(data: List[float], bins: Optional[int] = 10,
                   title: Optional[str] = 'Histogram',
                   xlabel: Optional[str] = 'Value',
                   ylabel: Optional[str] = 'Frequency') -> None:
    """
    Plots a histogram from a list of data points.

    :param data: List of data points to be plotted.
    :type data: List[float]
    :param bins: Number of bins for the histogram, defaults to 10.
    :type bins: Optional[int], optional
    :param title: Title of the histogram, defaults to 'Histogram'.
    :type title: Optional[str], optional
    :param xlabel: Label for the x-axis, defaults to 'Value'.
    :type xlabel: Optional[str], optional
    :param ylabel: Label for the y-axis, defaults to 'Frequency'.
    :type ylabel: Optional[str], optional
    :raises ValueError: If data list is empty or not a list of floats.
    """

    if not isinstance(data, list):
        raise ValueError("Data should be a list.")
    if not all(isinstance(i, (int, float)) for i in data):
        raise ValueError("All items in the data list must be int or float.")
    if len(data) == 0:
        raise ValueError("Data list cannot be empty.")

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

plot_histogram(times)
print(sum(times) / len(times))
