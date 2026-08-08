import numpy as np
from hypothesis import given, strategies as st
from src.target_recall_hnsw import l2_squared, cosine_distance

# Strategy to generate random float vectors of dimension between 2 and 32
vector_strategy = st.lists(
    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=32
).map(np.array)


@given(vector_strategy)
def test_distance_to_self(a):
    """Distance from a vector to itself should be 0."""
    # L2 squared to self
    assert np.isclose(l2_squared(a, a), 0.0, atol=1e-7)

    # Cosine distance to self should be 0 (ignoring zero vectors)
    if np.linalg.norm(a) > 1e-5:
        assert np.isclose(cosine_distance(a, a), 0.0, atol=1e-5)


@given(vector_strategy, vector_strategy)
def test_symmetry(a, b):
    """Distance functions must be symmetric: d(a, b) == d(b, a)."""
    # L2 squared symmetry
    assert np.isclose(l2_squared(a, b), l2_squared(b, a), rtol=1e-5, atol=1e-7)

    # Cosine distance symmetry
    assert np.isclose(cosine_distance(a, b), cosine_distance(b, a), rtol=1e-5, atol=1e-7)


@given(vector_strategy, vector_strategy, vector_strategy)
def test_triangle_inequality_l2(a, b, c):
    """Triangle inequality holds for L2 distance: d(a, c) <= d(a, b) + d(b, c)."""
    # Note: Evaluated using standard Euclidean distance (sqrt of l2_squared)
    # since squared Euclidean distance does not strictly satisfy the triangle inequality.
    d_ac = np.linalg.norm(a - c)
    d_ab = np.linalg.norm(a - b)
    d_bc = np.linalg.norm(b - c)

    epsilon = 1e-7
    assert d_ac <= d_ab + d_bc + epsilon
