"""Visualization functions for DDA results."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .results import CTResult, DEResult, STResult


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install 'dda-py[matplotlib]'"
        )


def _get_or_create_axes(
    ax: "Optional[Axes]", figsize: Tuple[float, float]
) -> "Tuple[Figure, Axes]":
    """Return (fig, ax), creating a new figure if ax is None."""
    plt = _require_matplotlib()
    if ax is not None:
        return ax.get_figure(), ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _get_x_axis(
    result: "Union[STResult, CTResult, DEResult]",
    use_time: bool,
    sfreq: Optional[float],
) -> np.ndarray:
    """Compute x-axis values."""
    if use_time:
        sf = sfreq or result.params.get("sfreq", 1.0)
        return result.window_starts / sf
    return np.arange(len(result.window_starts))


def _get_labels(result: "Union[STResult, CTResult]") -> List[str]:
    """Get channel or pair labels via duck typing."""
    if hasattr(result, "channel_labels"):
        return result.channel_labels
    return result.pair_labels


def plot_coefficients(
    result: "Union[STResult, CTResult]",
    coeff_indices: Optional[List[int]] = None,
    channels: Optional[List[int]] = None,
    use_time: bool = False,
    sfreq: Optional[float] = None,
    ax: "Optional[Axes]" = None,
    figsize: Tuple[float, float] = (10, 4),
) -> "Figure":
    """Plot time-series of DDA coefficients.

    Args:
        result: STResult or CTResult.
        coeff_indices: Which coefficients to plot (0-based). None = all.
        channels: Which channels/pairs to plot (0-based). None = all.
        use_time: If True, x-axis is time in seconds.
        sfreq: Sampling frequency for time axis.
        ax: Existing axes. If None, creates new figure.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    plt = _require_matplotlib()
    x = _get_x_axis(result, use_time, sfreq)
    labels = _get_labels(result)
    coeffs = result.coefficients  # (n_ch, n_win, n_coeff)

    if coeff_indices is None:
        coeff_indices = list(range(coeffs.shape[2]))
    if channels is None:
        channels = list(range(coeffs.shape[0]))

    n_plots = len(coeff_indices)
    if ax is not None:
        fig = ax.get_figure()
        axes = [ax] if n_plots == 1 else [ax]
    else:
        fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), squeeze=False)
        axes = axes[:, 0]

    for plot_idx, coeff_idx in enumerate(coeff_indices):
        cur_ax = axes[plot_idx] if plot_idx < len(axes) else axes[-1]
        for ch_idx in channels:
            cur_ax.plot(x, coeffs[ch_idx, :, coeff_idx], label=labels[ch_idx])
        cur_ax.set_ylabel(f"a_{coeff_idx + 1}")
        cur_ax.legend(fontsize="small")

    axes[-1].set_xlabel("Time (s)" if use_time else "Window")
    fig.tight_layout()
    return fig


def plot_errors(
    result: "Union[STResult, CTResult]",
    channels: Optional[List[int]] = None,
    use_time: bool = False,
    sfreq: Optional[float] = None,
    ax: "Optional[Axes]" = None,
    figsize: Tuple[float, float] = (10, 4),
) -> "Figure":
    """Plot time-series of fitting errors.

    Args:
        result: STResult or CTResult.
        channels: Which channels to plot (0-based). None = all.
        use_time: If True, x-axis is time in seconds.
        sfreq: Sampling frequency.
        ax: Existing axes.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = _get_or_create_axes(ax, figsize)
    x = _get_x_axis(result, use_time, sfreq)
    labels = _get_labels(result)

    if channels is None:
        channels = list(range(result.errors.shape[0]))

    for ch_idx in channels:
        ax.plot(x, result.errors[ch_idx], label=labels[ch_idx])

    ax.set_xlabel("Time (s)" if use_time else "Window")
    ax.set_ylabel("Error")
    ax.legend(fontsize="small")
    fig.tight_layout()
    return fig


def plot_heatmap(
    result: "Union[STResult, CTResult]",
    coeff_index: int = 0,
    use_time: bool = False,
    sfreq: Optional[float] = None,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: "Optional[Axes]" = None,
    figsize: Tuple[float, float] = (12, 6),
) -> "Figure":
    """Plot channels x windows heatmap for a selected coefficient.

    Args:
        result: STResult or CTResult.
        coeff_index: Which coefficient (0-based, e.g. 0 for a_1).
        use_time: If True, x-axis is time.
        sfreq: Sampling frequency.
        cmap: Colormap name.
        vmin, vmax: Color range bounds.
        ax: Existing axes.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    plt = _require_matplotlib()
    fig, ax = _get_or_create_axes(ax, figsize)

    data = result.coefficients[:, :, coeff_index]  # (n_ch, n_win)
    labels = _get_labels(result)
    x = _get_x_axis(result, use_time, sfreq)

    if vmin is None and vmax is None:
        abs_max = np.abs(data).max()
        vmin, vmax = -abs_max, abs_max

    extent = [x[0], x[-1], len(labels) - 0.5, -0.5]
    im = ax.imshow(
        data,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)" if use_time else "Window")
    ax.set_ylabel("Channel")
    ax.set_title(f"a_{coeff_index + 1}")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_ergodicity(
    result: "DEResult",
    use_time: bool = False,
    sfreq: Optional[float] = None,
    ax: "Optional[Axes]" = None,
    figsize: Tuple[float, float] = (10, 4),
) -> "Figure":
    """Plot DE ergodicity time-series.

    Args:
        result: DEResult.
        use_time: If True, x-axis is time.
        sfreq: Sampling frequency.
        ax: Existing axes.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = _get_or_create_axes(ax, figsize)
    x = _get_x_axis(result, use_time, sfreq)

    ax.plot(x, result.ergodicity, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Time (s)" if use_time else "Window")
    ax.set_ylabel("Ergodicity")
    ax.set_title("Dynamical Ergodicity")
    fig.tight_layout()
    return fig


def plot_model(
    model_encoding: List[int],
    num_delays: int = 2,
    polynomial_order: int = 4,
    ax: "Optional[Axes]" = None,
    figsize: Tuple[float, float] = (8, 6),
) -> "Figure":
    """Visualize model space grid showing selected/unselected terms.

    Draws a grid where each cell is a monomial. Selected terms are
    highlighted, unselected are greyed out.

    Args:
        model_encoding: 1-based indices of selected monomials.
        num_delays: Number of delay values.
        polynomial_order: Maximum polynomial degree.
        ax: Existing axes.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    from .model_encoding import generate_monomials, monomial_to_text

    plt = _require_matplotlib()
    fig, ax = _get_or_create_axes(ax, figsize)

    monomials = generate_monomials(num_delays, polynomial_order)
    selected_set = set(model_encoding)

    # Group by effective degree (number of non-zero factors)
    degree_groups: dict[int, list[tuple[int, tuple, str]]] = {}
    for idx, mono in enumerate(monomials, start=1):
        degree = sum(1 for v in mono if v > 0)
        text = monomial_to_text(mono)
        degree_groups.setdefault(degree, []).append((idx, mono, text))

    degrees = sorted(degree_groups.keys())
    max_per_row = max(len(v) for v in degree_groups.values())

    ax.set_xlim(-0.5, max_per_row - 0.5)
    ax.set_ylim(-0.5, len(degrees) - 0.5)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks(range(len(degrees)))
    ax.set_yticklabels([f"degree {d}" for d in degrees])

    for row, degree in enumerate(degrees):
        items = degree_groups[degree]
        for col, (idx, mono, text) in enumerate(items):
            is_selected = idx in selected_set
            color = "#4CAF50" if is_selected else "#E0E0E0"
            text_color = "white" if is_selected else "#999"
            fontweight = "bold" if is_selected else "normal"

            rect = plt.Rectangle(
                (col - 0.4, row - 0.35),
                0.8,
                0.7,
                facecolor=color,
                edgecolor="#666",
                linewidth=0.5,
                zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
                fontweight=fontweight,
                zorder=3,
            )

    ax.set_title(
        f"Model Space ({num_delays} delays, order {polynomial_order})"
    )
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
