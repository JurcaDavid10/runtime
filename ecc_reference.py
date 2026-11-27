
"""Classical reference implementation for modulo-17 ECC workloads."""

MODULUS = 17
CURVE_A = 2
CURVE_B = 2
CURVE_ORDER = 19

Point = tuple[int, int] | None

GENERATOR: Point = (5, 1)
PUBLIC_KEY: Point = (0, 6)
POINT_AT_INFINITY: Point = None


# ============================================================
# FINITE-FIELD ARITHMETIC
# ============================================================

def mod_add(a: int, b: int) -> int:
    return (a + b) % MODULUS


def mod_sub(a: int, b: int) -> int:
    return (a - b) % MODULUS


def mod_mul(a: int, b: int) -> int:
    return (a * b) % MODULUS


def mod_square(a: int) -> int:
    return (a * a) % MODULUS


def mod_inv(a: int) -> int:
    a %= MODULUS

    if a == 0:
        raise ZeroDivisionError(
            "Zero has no modular inverse."
        )

    return pow(a, -1, MODULUS)


# ============================================================
# ELLIPTIC-CURVE OPERATIONS
# ============================================================

def is_on_curve(point: Point) -> bool:
    if point is None:
        return True

    x, y = point

    if not (0 <= x < MODULUS and 0 <= y < MODULUS):
        return False

    lhs = mod_square(y)

    rhs = (
        x**3 + CURVE_A * x + CURVE_B
    ) % MODULUS

    return lhs == rhs


def point_neg(point: Point) -> Point:
    if not is_on_curve(point):
        raise ValueError("Invalid elliptic-curve point.")

    if point is None:
        return None

    x, y = point

    return x, (-y) % MODULUS


def point_add(p: Point, q: Point) -> Point:
    if not is_on_curve(p) or not is_on_curve(q):
        raise ValueError("Invalid elliptic-curve point.")

    if p is None:
        return q

    if q is None:
        return p

    x1, y1 = p
    x2, y2 = q

    # Inverse points or doubling with y = 0.
    if x1 == x2 and mod_add(y1, y2) == 0:
        return None

    if p == q:
        # Point doubling.
        numerator = mod_add(
            mod_mul(3, mod_square(x1)),
            CURVE_A,
        )
        denominator = mod_mul(2, y1)

    else:
        # Addition of distinct points.
        numerator = mod_sub(y2, y1)
        denominator = mod_sub(x2, x1)

    slope = mod_mul(
        numerator,
        mod_inv(denominator),
    )

    x3 = mod_sub(
        mod_sub(mod_square(slope), x1),
        x2,
    )

    y3 = mod_sub(
        mod_mul(slope, mod_sub(x1, x3)),
        y1,
    )

    return x3, y3


def point_double(point: Point) -> Point:
    return point_add(point, point)


def scalar_mul(k: int, point: Point) -> Point:
    """Scalar multiplication using double-and-add."""

    if not is_on_curve(point):
        raise ValueError("Invalid elliptic-curve point.")

    if k < 0:
        return scalar_mul(-k, point_neg(point))

    result = POINT_AT_INFINITY
    addend = point

    while k > 0:
        if k & 1:
            result = point_add(result, addend)

        addend = point_double(addend)
        k >>= 1

    return result


def enumerate_curve_points() -> list[Point]:
    points: list[Point] = [POINT_AT_INFINITY]

    for x in range(MODULUS):
        for y in range(MODULUS):
            if is_on_curve((x, y)):
                points.append((x, y))

    return points


# ============================================================
# REFERENCE DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QARW - CLASSICAL ECC REFERENCE")
    print("=" * 60)

    print(f"Modulus: {MODULUS}")
    print(f"Curve: y^2 = x^3 + {CURVE_A}x + {CURVE_B}")
    print(f"Generator: {GENERATOR}")
    print(f"Public key: {PUBLIC_KEY}")
    print(f"Curve order: {len(enumerate_curve_points())}")

    print("\nFINITE-FIELD OPERATIONS")
    print(f"13 + 9 mod 17 = {mod_add(13, 9)}")
    print(f"3 - 8 mod 17 = {mod_sub(3, 8)}")
    print(f"5 * 7 mod 17 = {mod_mul(5, 7)}")
    print(f"5^2 mod 17 = {mod_square(5)}")
    print(f"5^(-1) mod 17 = {mod_inv(5)}")

    print("\nELLIPTIC-CURVE OPERATIONS")
    print(f"2G = {point_double(GENERATOR)}")
    print(f"7G = {scalar_mul(7, GENERATOR)}")
    print(f"19G = {scalar_mul(19, GENERATOR)}")

    print("\nSCALAR MULTIPLICATION TABLE")

    for k in range(1, CURVE_ORDER + 1):
        print(f"{k:2d}G = {scalar_mul(k, GENERATOR)}")

    assert scalar_mul(7, GENERATOR) == PUBLIC_KEY
    assert scalar_mul(CURVE_ORDER, GENERATOR) is None
    assert len(enumerate_curve_points()) == CURVE_ORDER

    print("\nPASS: Classical ECC reference validation")
