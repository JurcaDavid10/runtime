
import pytest

from ecc_reference import (
    MODULUS,
    CURVE_ORDER,
    GENERATOR,
    PUBLIC_KEY,
    mod_add,
    mod_sub,
    mod_mul,
    mod_square,
    mod_inv,
    is_on_curve,
    point_neg,
    point_add,
    point_double,
    scalar_mul,
    enumerate_curve_points,
)


def test_modular_arithmetic():
    for a in range(MODULUS):
        assert mod_square(a) == (a * a) % MODULUS

        for b in range(MODULUS):
            assert mod_add(a, b) == (a + b) % MODULUS
            assert mod_sub(a, b) == (a - b) % MODULUS
            assert mod_mul(a, b) == (a * b) % MODULUS


def test_modular_inverse():
    for a in range(1, MODULUS):
        inverse = mod_inv(a)
        assert mod_mul(a, inverse) == 1

    with pytest.raises(ZeroDivisionError):
        mod_inv(0)


def test_curve_order():
    points = enumerate_curve_points()

    assert len(points) == CURVE_ORDER
    assert len(set(points)) == CURVE_ORDER
    assert GENERATOR in points
    assert PUBLIC_KEY in points


def test_known_points():
    assert is_on_curve(GENERATOR)
    assert is_on_curve(PUBLIC_KEY)
    assert is_on_curve(None)

    assert point_double(GENERATOR) == (6, 3)
    assert scalar_mul(3, GENERATOR) == (10, 6)
    assert scalar_mul(7, GENERATOR) == PUBLIC_KEY

    assert not is_on_curve((1, 1))


def test_identity_and_inverse():
    for point in enumerate_curve_points():
        assert point_add(point, None) == point
        assert point_add(None, point) == point
        assert point_add(point, point_neg(point)) is None


def test_group_closure_and_commutativity():
    points = enumerate_curve_points()

    for p in points:
        for q in points:
            result = point_add(p, q)

            assert result in points
            assert result == point_add(q, p)


def test_group_associativity():
    points = enumerate_curve_points()

    for p in points:
        for q in points:
            for r in points:
                assert point_add(point_add(p, q), r) == (
                    point_add(p, point_add(q, r))
                )


def test_scalar_multiplication():
    result = None
    generated_points = set()

    for k in range(CURVE_ORDER):
        assert scalar_mul(k, GENERATOR) == result

        generated_points.add(result)
        result = point_add(result, GENERATOR)

    assert generated_points == set(enumerate_curve_points())
    assert scalar_mul(CURVE_ORDER, GENERATOR) is None

    assert scalar_mul(-1, GENERATOR) == point_neg(GENERATOR)

    assert scalar_mul(
        CURVE_ORDER + 7, GENERATOR
    ) == PUBLIC_KEY


def test_invalid_curve_points():
    with pytest.raises(ValueError):
        point_add((1, 1), GENERATOR)

    with pytest.raises(ValueError):
        scalar_mul(2, (1, 1))
