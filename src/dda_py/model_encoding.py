"""
DDA Model Encoding Utilities

This module provides utilities for visualizing and decoding DDA MODEL parameter encodings
into their corresponding delay differential equation (DDE) representations.

The MODEL parameter encodes which polynomial terms to include in the DDE model.
For a given number of delays and polynomial order, all possible monomials are enumerated
and assigned 1-based indices. The MODEL encoding then selects specific monomials by index.

Monomial Generation (Canonical Format):
    All non-decreasing tuples of length `polynomial_order` with values in
    {0, 1, ..., num_delays}, excluding the all-zeros tuple. Value 0 means
    "no factor" (padding), value k > 0 means x(t - tau_k).

Example:
    For 2 delays and polynomial order 4, the monomial space includes:
    - Index 1:  (0, 0, 0, 1) -> x(t-tau_1)
    - Index 2:  (0, 0, 0, 2) -> x(t-tau_2)
    - Index 3:  (0, 0, 1, 1) -> x(t-tau_1)^2
    - ...
    - Index 10: (1, 1, 1, 1) -> x(t-tau_1)^4
    - ...
    - Index 14: (2, 2, 2, 2) -> x(t-tau_2)^4

    Model encoding [1, 2, 10] represents: dx/dt = a_1 x(t-tau_1) + a_2 x(t-tau_2) + a_3 x(t-tau_1)^4
"""

from typing import List, Tuple, Generator
from collections import Counter


def _non_decreasing_tuples(
    length: int, min_val: int, max_val: int
) -> Generator[Tuple[int, ...], None, None]:
    """Generate all non-decreasing tuples of given length with values in [min_val, max_val].

    Args:
        length: Length of each tuple.
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).

    Yields:
        Non-decreasing tuples of the specified length.
    """
    if length == 0:
        yield ()
        return
    for v in range(min_val, max_val + 1):
        for suffix in _non_decreasing_tuples(length - 1, v, max_val):
            yield (v,) + suffix


def generate_monomials(num_delays: int, polynomial_order: int) -> List[Tuple[int, ...]]:
    """Generate all monomial encodings for DDA model space.

    Generates all non-decreasing tuples of length ``polynomial_order`` with values
    in {0, 1, ..., num_delays}, excluding the all-zeros tuple. This matches the
    canonical format used by the DDA binary and the TypeScript/Rust implementations.

    Each tuple has fixed length equal to ``polynomial_order``. Value 0 means
    "no factor" (padding), value k > 0 means x(t - tau_k).

    Args:
        num_delays: Number of delay values (tau values).
        polynomial_order: Maximum polynomial degree (determines tuple length).

    Returns:
        List of tuples representing monomials in canonical order.

    Example:
        >>> generate_monomials(2, 2)
        [(0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

        >>> generate_monomials(2, 3)
        [(0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 1, 2), (0, 2, 2),
         (1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2)]
    """
    all_zeros = (0,) * polynomial_order
    return [
        t for t in _non_decreasing_tuples(polynomial_order, 0, num_delays)
        if t != all_zeros
    ]


def _group_factors(monomial: Tuple[int, ...]) -> Counter:
    """Count non-zero factors in a monomial encoding.

    Args:
        monomial: Canonical monomial tuple (may contain leading zeros).

    Returns:
        Counter mapping delay index -> exponent count.
    """
    return Counter(v for v in monomial if v > 0)


def monomial_to_latex(monomial: Tuple[int, ...], tau_values: List[float] = None) -> str:
    """Convert a monomial encoding to LaTeX representation.

    Args:
        monomial: Canonical tuple representing the monomial.
        tau_values: Optional list of delay values to include in notation.

    Returns:
        LaTeX string representation.

    Example:
        >>> monomial_to_latex((0, 0, 0, 1))
        'x_{1}'
        >>> monomial_to_latex((0, 0, 1, 1))
        'x_{1}^{2}'
        >>> monomial_to_latex((0, 0, 1, 2))
        'x_{1} x_{2}'
        >>> monomial_to_latex((0, 1), tau_values=[7, 10])
        'x(t - 7)'
    """
    counts = _group_factors(monomial)
    if not counts:
        return "1"

    terms = []
    for idx in sorted(counts.keys()):
        if tau_values is not None and idx <= len(tau_values):
            tau = tau_values[idx - 1]
            base = f"x(t - {tau})"
            if counts[idx] > 1:
                terms.append(f"{base}^{{{counts[idx]}}}")
            else:
                terms.append(base)
        else:
            if counts[idx] > 1:
                terms.append(f"x_{{{idx}}}^{{{counts[idx]}}}")
            else:
                terms.append(f"x_{{{idx}}}")

    return " ".join(terms)


def monomial_to_text(monomial: Tuple[int, ...], tau_values: List[float] = None) -> str:
    """Convert a monomial encoding to plain text representation.

    Args:
        monomial: Canonical tuple representing the monomial.
        tau_values: Optional list of delay values to include in notation.

    Returns:
        Plain text string representation.

    Example:
        >>> monomial_to_text((0, 0, 0, 1))
        'x_1'
        >>> monomial_to_text((0, 0, 1, 1))
        'x_1^2'
        >>> monomial_to_text((0, 0, 1, 2))
        'x_1 * x_2'
    """
    counts = _group_factors(monomial)
    if not counts:
        return "1"

    terms = []
    for idx in sorted(counts.keys()):
        if tau_values is not None and idx <= len(tau_values):
            tau = tau_values[idx - 1]
            base = f"x(t - {tau})"
            if counts[idx] > 1:
                terms.append(f"{base}^{counts[idx]}")
            else:
                terms.append(base)
        else:
            if counts[idx] > 1:
                terms.append(f"x_{idx}^{counts[idx]}")
            else:
                terms.append(f"x_{idx}")

    return " * ".join(terms)


def decode_model_encoding(
    model_encoding: List[int],
    num_delays: int,
    polynomial_order: int,
    tau_values: List[float] = None,
    format: str = "latex"
) -> str:
    """Decode a MODEL parameter into its DDE representation.

    Args:
        model_encoding: List of 1-based indices selecting monomials (e.g., [1, 2, 10]).
        num_delays: Number of delay values.
        polynomial_order: Maximum polynomial degree.
        tau_values: Optional list of actual tau values to include in notation.
        format: Output format ("latex" or "text").

    Returns:
        String representation of the DDE equation.

    Raises:
        ValueError: If any index in model_encoding is out of range.

    Example:
        >>> decode_model_encoding([1, 3, 5], 2, 2, format="text")
        'dx/dt = a_1 x_1 + a_2 x_1^2 + a_3 x_2^2'
    """
    monomials = generate_monomials(num_delays, polynomial_order)

    terms = []
    for coeff_idx, monomial_idx in enumerate(model_encoding, start=1):
        if monomial_idx < 1 or monomial_idx > len(monomials):
            raise ValueError(
                f"Invalid model index {monomial_idx}. "
                f"Must be in range [1, {len(monomials)}]"
            )

        monomial = monomials[monomial_idx - 1]

        if format == "latex":
            coeff_str = f"a_{{{coeff_idx}}}"
            term_str = monomial_to_latex(monomial, tau_values)
            terms.append(f"{coeff_str} {term_str}")
        else:
            coeff_str = f"a_{coeff_idx}"
            term_str = monomial_to_text(monomial, tau_values)
            terms.append(f"{coeff_str} {term_str}")

    equation = " + ".join(terms)

    if format == "latex":
        return f"\\dot{{x}} = {equation}"
    else:
        return f"dx/dt = {equation}"


def visualize_model_space(
    num_delays: int,
    polynomial_order: int,
    tau_values: List[float] = None,
    highlight_encoding: List[int] = None
) -> str:
    """Visualize the complete model space with all available monomials.

    Args:
        num_delays: Number of delay values.
        polynomial_order: Maximum polynomial degree.
        tau_values: Optional list of actual tau values to include in output.
        highlight_encoding: Optional model encoding to highlight selected terms.

    Returns:
        Formatted string showing all monomials with their indices.

    Example:
        >>> print(visualize_model_space(2, 2))
        Model Space: 2 delays, order 2
        Total monomials: 5
        <BLANKLINE>
        Index | Encoding    | Term
        ------|-------------|---------------------
           1 | (0, 1)      | x_1
           2 | (0, 2)      | x_2
           3 | (1, 1)      | x_1^2
           4 | (1, 2)      | x_1 * x_2
           5 | (2, 2)      | x_2^2
    """
    monomials = generate_monomials(num_delays, polynomial_order)

    header_parts = [f"Model Space: {num_delays} delays, order {polynomial_order}"]
    if tau_values:
        tau_str = ", ".join(f"τ_{i+1}={v}" for i, v in enumerate(tau_values))
        header_parts.append(f"Delays: {tau_str}")
    header_parts.append(f"Total monomials: {len(monomials)}")

    lines = header_parts + [
        "",
        "Index | Encoding    | Term",
        "------|-------------|---------------------"
    ]

    highlight_set = set(highlight_encoding) if highlight_encoding else set()

    for idx, monomial in enumerate(monomials, start=1):
        encoding_str = str(monomial)
        term_str = monomial_to_text(monomial, tau_values)

        marker = " *" if idx in highlight_set else "  "
        lines.append(f"{marker}{idx:4d} | {encoding_str:11s} | {term_str}")

    if highlight_encoding:
        lines.extend([
            "",
            f"Selected terms (marked with *): {highlight_encoding}",
            decode_model_encoding(
                highlight_encoding,
                num_delays,
                polynomial_order,
                tau_values,
                format="text"
            )
        ])

    return "\n".join(lines)


def model_encoding_to_dict(
    model_encoding: List[int],
    num_delays: int,
    polynomial_order: int,
    tau_values: List[float] = None
) -> dict:
    """Convert model encoding to structured dictionary representation.

    Useful for API responses or JSON serialization.

    Args:
        model_encoding: List of 1-based indices.
        num_delays: Number of delay values.
        polynomial_order: Maximum polynomial degree.
        tau_values: Optional list of actual tau values.

    Returns:
        Dictionary with equation details.

    Example:
        >>> model_encoding_to_dict([1, 3, 5], 2, 2)
        {'equation_latex': '...', 'equation_text': '...', 'num_terms': 3, ...}
    """
    monomials = generate_monomials(num_delays, polynomial_order)

    terms = []
    for coeff_idx, monomial_idx in enumerate(model_encoding, start=1):
        if monomial_idx < 1 or monomial_idx > len(monomials):
            raise ValueError(f"Invalid model index {monomial_idx}")

        monomial = monomials[monomial_idx - 1]

        terms.append({
            'coefficient': f'a_{coeff_idx}',
            'monomial_index': monomial_idx,
            'monomial': list(monomial),
            'term_text': monomial_to_text(monomial, tau_values),
            'term_latex': monomial_to_latex(monomial, tau_values)
        })

    return {
        'equation_latex': decode_model_encoding(
            model_encoding, num_delays, polynomial_order, tau_values, format="latex"
        ),
        'equation_text': decode_model_encoding(
            model_encoding, num_delays, polynomial_order, tau_values, format="text"
        ),
        'num_delays': num_delays,
        'polynomial_order': polynomial_order,
        'tau_values': tau_values,
        'num_terms': len(terms),
        'terms': terms
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m dda_py.model_encoding <num_delays> <polynomial_order> [model_encoding...]")
        print("\nExample:")
        print("  python -m dda_py.model_encoding 2 2")
        print("  python -m dda_py.model_encoding 2 4 1 2 10")
        sys.exit(1)

    num_delays = int(sys.argv[1])
    poly_order = int(sys.argv[2])

    if len(sys.argv) > 3:
        model_enc = [int(x) for x in sys.argv[3:]]
        print(visualize_model_space(num_delays, poly_order, highlight_encoding=model_enc))
    else:
        print(visualize_model_space(num_delays, poly_order))
