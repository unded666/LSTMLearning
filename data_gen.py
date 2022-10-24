import numpy as np
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import square, sawtooth


def generate_sine(x_in: np.array,
                  period: float,
                  amplitude: float,
                  ) -> np.array:
    """
    generates a sinusoid as defined

    :param x_in: input x values
    :param period: periodicity of each wave within the waveform
    :param amplitude: A. self-explanatory
    :return: output sinusoid
    """

    out = np.sin(x_in * 2 * np.pi / period) * amplitude

    return out


def generate_square(x_in: np.array,
                    period: float,
                    amplitude: float,
                    ) -> float:
    """
    generates a square wave as defined

    :param x_in: input x values
    :param period: periodicity of each wave within the waveform
    :param amplitude: Height of the square wave
    :return: output square wave
    """

    out = square(2 * np.pi * x_in / period) * amplitude

    return out


def generate_sawtooth(x_in: np.array,
                      period: float,
                      amplitude: float,
                      ) -> float:
    """
    generates a sawtooth wave as defined

    :param x_in: input x values
    :param period: periodicity of each wave within the waveform
    :param amplitude: Height of the square wave
    :return: output square wave
    """

    out = sawtooth(2 * np.pi * x_in / period, 1) * amplitude

    return out


def generate_triangle(x_in: np.array,
                      period: float,
                      amplitude: float,
                      ) -> float:
    """
    generates a triangle wave as defined

    :param x_in: input x values
    :param period: periodicity of each wave within the waveform
    :param amplitude: Height of the square wave
    :return: output square wave
    """

    out = sawtooth(2 * np.pi * x_in / period, 0.5) * amplitude

    return out


class WaveGenerator:

    class WaveType(Enum):
        SINUSOID = 1
        SQUARE = 2
        SAWTOOTH = 3
        TRIANGLE = 4

    def __init__(self,
                 waveform: WaveType = WaveType.SINUSOID,
                 data_range: list = (-5, 5),
                 periodicity: float = 2.5,
                 sample_freq: float = 0.01,
                 amplitude: float = 1,
                 noise_percent: float = 0.05,
                 ) -> None:
        """

        :param waveform: WaveType, can be sine, square, sawtooth or triangle
        :param data_range: minimum and maximum x-values for waveform f(x)
        :param periodicity: T, period for waveform
        :param sample_freq: sampling frequency, aka frequency of x-values
        :param amplitude: amplitude A of the waveform
        :param noise_percent: % random noise to be added to the waveform
        """

        self.waveform = waveform
        self.data_range = data_range
        self.periodicity = periodicity
        self.amplitude = amplitude
        self.noise_percent = noise_percent
        self.sample_freq = sample_freq
        self.waveform_data = pd.DataFrame(columns=['x', 'y'])

    def generate_x_values(self) -> None:
        """
        generates x_values from the object parameters, stores them in the
        waveform_data dataframe
        :return: None
        """

        x_values = np.append(np.arange(self.data_range[0], self.data_range[1], self.sample_freq), self.data_range[1])
        self.waveform_data['x'] = x_values

    def generate_y_values(self) -> None:
        """
        does the hard work of creating the waveforms. four potential waveforms, dependant on the
        value of the waveform parameter
        :return: None
        """

        function_dict = {self.WaveType.SINUSOID: generate_sine,
                         self.WaveType.SQUARE: generate_square,
                         self.WaveType.SAWTOOTH: generate_sawtooth,
                         self.WaveType.TRIANGLE: generate_triangle}
        function = function_dict[self.waveform]
        y_values = function(self.waveform_data['x'], self.periodicity, self.amplitude)
        self.waveform_data['y'] = y_values

    def generate_noise(self) -> None:
        """
        adds noise to the generated y values
        :return:
        """

        num_points = self.waveform_data.shape[0]
        random_values = np.random.random(num_points) - 0.5
        random_values = random_values * self.amplitude * self.noise_percent * 2
        self.waveform_data.y = self.waveform_data.y + random_values

    def data_generation(self) -> (np.array, np.array):
        """
        generates the appropriate x and y values, and then returns them as
        numpy arrays
        :return: x, y
        """

        self.generate_x_values()
        self.generate_y_values()
        self.generate_noise()

        return self.waveform_data.x.to_numpy().copy(), self.waveform_data.y.to_numpy().copy()


if __name__ == '__main__':
    # debugging code

    generator = WaveGenerator(periodicity=4,amplitude=5)
    generator.generate_x_values()
    generator.generate_y_values()
    plt.plot(generator.waveform_data.x, generator.waveform_data.y, label = 'Clean Data')
    plt.grid()
    plt.xticks(np.linspace(-5, 5, 11))
    generator.generate_noise()
    plt.plot(generator.waveform_data.x, generator.waveform_data.y, label = 'Noisy Data')
    plt.legend()
