from unittest import TestCase
from data_gen import (
    WaveGenerator,
    generate_sine,
    generate_square,
    generate_sawtooth,
    generate_triangle,
)

import numpy as np
import pandas as pd

class WaveTest(TestCase):
    def setUp(self) -> None:
        """
        creates a number of waveform test objects
        :return:None
        """

        self.sine = WaveGenerator(waveform=WaveGenerator.WaveType.SINUSOID,
                                  data_range=(-5, 5),
                                  periodicity=2.5,
                                  sample_freq=0.01,
                                  amplitude=1,
                                  noise_percent=0.05,
                                  )
        self.square = WaveGenerator(waveform=WaveGenerator.WaveType.SQUARE,
                                    data_range=(-5, 5),
                                    periodicity=2.5,
                                    sample_freq=0.01,
                                    amplitude=3,
                                    noise_percent=0.05,
                                    )
        self.tri = WaveGenerator(waveform=WaveGenerator.WaveType.TRIANGLE,
                                 data_range=(-5, 5),
                                 periodicity=2.5,
                                 sample_freq=0.01,
                                 amplitude=2,
                                 noise_percent=0.05,
                                 )
        self.sawtooth = WaveGenerator(waveform=WaveGenerator.WaveType.SAWTOOTH,
                                      data_range=(-5, 5),
                                      periodicity=2.5,
                                      sample_freq=0.01,
                                      amplitude=4,
                                      noise_percent=0.05,
                                      )

    def test_generate_x_values(self) -> None:

        self.sine.generate_x_values()
        self.assertEqual(self.sine.waveform_data.shape[0], 1001, 'incorrect number of x data points created')

    def test_generate_sine(self):

        self.sine.generate_x_values()
        y = generate_sine(self.sine.waveform_data['x'].to_numpy(),
                          self.sine.periodicity,
                          self.sine.amplitude)
        round_y = np.round(y, 3)
        peak_values = np.argwhere(round_y == 1)
        peak_targets = [[62], [63], [312], [313], [562], [563], [812], [813]]
        self.assertCountEqual(peak_targets, peak_values, 'incorrect sine wave created')

    def test_generate_square(self):

        self.square.generate_x_values()
        y = generate_square(self.square.waveform_data['x'].to_numpy(),
                            self.square.periodicity,
                            self.square.amplitude)
        IND = [2, 200, 405, 700]
        test_y = y[IND]
        targets = [3, -3, -3, -3]
        self.assertCountEqual(test_y, targets, 'incorrect square wave generated')

    def test_generate_sawtooth(self):

        self.sawtooth.generate_x_values()
        y = generate_sawtooth(self.sawtooth.waveform_data['x'].to_numpy(),
                              self.sawtooth.periodicity,
                              self.sawtooth.amplitude)
        round_y = np.round(y, 3)
        peak_values = np.argwhere(round_y == 4)
        targets = [[250], [500], [750]]
        self.assertCountEqual(peak_values, targets, 'sawtooth incorrectly generated')

    def test_generate_triangle(self):

        self.tri.generate_x_values()
        y = generate_triangle(self.tri.waveform_data['x'].to_numpy(),
                              self.tri.periodicity,
                              self.tri.amplitude)
        round_y = np.round(y, 3)
        peak_values = np.argwhere(round_y == 2)
        targets = [[125], [375], [625], [875]]
        self.assertCountEqual(peak_values, targets, 'triangle incorrectly generated')

    def test_random_noise(self):

        self.sine.generate_x_values()
        self.sine.generate_y_values()
        clean_y = self.sine.waveform_data.y.to_numpy()
        self.sine.generate_noise()
        noisy_y = self.sine.waveform_data.y.to_numpy()
        differences = np.abs(clean_y - noisy_y)
        max_diff = self.sine.amplitude*self.sine.noise_percent
        excessive = differences > max_diff
        self.assertEqual(np.sum(excessive), 0, 'random noise exceeds bounds')







