"""Tests for plotting module."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from matplotlib.figure import Figure

from dda_py.plotting import (
    plot_coefficients,
    plot_ergodicity,
    plot_errors,
    plot_heatmap,
    plot_model,
)


class TestPlotCoefficients:
    def test_returns_figure(self, mock_st_result):
        fig = plot_coefficients(mock_st_result())
        assert isinstance(fig, Figure)

    def test_with_existing_axes(self, mock_st_result):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_fig = plot_coefficients(mock_st_result(), ax=ax)
        assert isinstance(result_fig, Figure)
        plt.close("all")

    def test_coeff_indices_filter(self, mock_st_result):
        fig = plot_coefficients(mock_st_result(), coeff_indices=[0])
        assert isinstance(fig, Figure)

    def test_channel_filter(self, mock_st_result):
        fig = plot_coefficients(mock_st_result(), channels=[0, 1])
        assert isinstance(fig, Figure)

    def test_use_time_axis(self, mock_st_result):
        fig = plot_coefficients(mock_st_result(), use_time=True, sfreq=256.0)
        assert isinstance(fig, Figure)

    def test_ct_result(self, mock_ct_result):
        fig = plot_coefficients(mock_ct_result())
        assert isinstance(fig, Figure)


class TestPlotErrors:
    def test_returns_figure(self, mock_st_result):
        fig = plot_errors(mock_st_result())
        assert isinstance(fig, Figure)

    def test_with_ax(self, mock_st_result):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_fig = plot_errors(mock_st_result(), ax=ax)
        assert isinstance(result_fig, Figure)
        plt.close("all")

    def test_channel_filter(self, mock_st_result):
        fig = plot_errors(mock_st_result(), channels=[1])
        assert isinstance(fig, Figure)


class TestPlotHeatmap:
    def test_returns_figure(self, mock_st_result):
        fig = plot_heatmap(mock_st_result())
        assert isinstance(fig, Figure)

    def test_custom_cmap(self, mock_st_result):
        fig = plot_heatmap(mock_st_result(), cmap="viridis")
        assert isinstance(fig, Figure)

    def test_coeff_index(self, mock_st_result):
        fig = plot_heatmap(mock_st_result(), coeff_index=2)
        assert isinstance(fig, Figure)

    def test_ct_result(self, mock_ct_result):
        fig = plot_heatmap(mock_ct_result())
        assert isinstance(fig, Figure)

    def test_custom_vlim(self, mock_st_result):
        fig = plot_heatmap(mock_st_result(), vmin=-1, vmax=1)
        assert isinstance(fig, Figure)

    def test_use_time(self, mock_st_result):
        fig = plot_heatmap(mock_st_result(), use_time=True)
        assert isinstance(fig, Figure)


class TestPlotErgodicity:
    def test_returns_figure(self, mock_de_result):
        fig = plot_ergodicity(mock_de_result())
        assert isinstance(fig, Figure)

    def test_with_time_axis(self, mock_de_result):
        fig = plot_ergodicity(mock_de_result(), use_time=True, sfreq=256.0)
        assert isinstance(fig, Figure)

    def test_with_ax(self, mock_de_result):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_fig = plot_ergodicity(mock_de_result(), ax=ax)
        assert isinstance(result_fig, Figure)
        plt.close("all")


class TestPlotModel:
    def test_returns_figure(self):
        fig = plot_model([1, 2, 10])
        assert isinstance(fig, Figure)

    def test_with_ax(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result_fig = plot_model([1, 2, 10], ax=ax)
        assert isinstance(result_fig, Figure)
        plt.close("all")

    def test_custom_params(self):
        fig = plot_model([1, 3, 5], num_delays=2, polynomial_order=2)
        assert isinstance(fig, Figure)

    def test_default_params(self):
        fig = plot_model([1, 2, 10], num_delays=2, polynomial_order=4)
        assert isinstance(fig, Figure)
