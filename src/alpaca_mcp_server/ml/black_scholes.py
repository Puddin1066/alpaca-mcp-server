from __future__ import annotations

import math
from dataclasses import dataclass


def _norm_cdf(x: float) -> float:
    # Standard normal CDF via erf; avoids scipy dependency.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class BlackScholesInputs:
    spot: float  # S
    strike: float  # K
    time_to_expiry_years: float  # T
    rate: float  # r (annualized, continuously compounded)
    volatility: float  # sigma (annualized)
    option_type: str  # "call" or "put"


def black_scholes_price(i: BlackScholesInputs) -> float:
    """
    Black-Scholes European option price.

    Notes:
    - Uses continuous compounding for r.
    - Assumes no dividends (or they are embedded in spot via forward adjustment).
    """
    if i.time_to_expiry_years <= 0:
        intrinsic = max(0.0, i.spot - i.strike) if i.option_type == "call" else max(0.0, i.strike - i.spot)
        return intrinsic
    if i.volatility <= 0:
        intrinsic = max(0.0, i.spot - i.strike) if i.option_type == "call" else max(0.0, i.strike - i.spot)
        return intrinsic * math.exp(-i.rate * i.time_to_expiry_years)

    s = float(i.spot)
    k = float(i.strike)
    t = float(i.time_to_expiry_years)
    r = float(i.rate)
    sig = float(i.volatility)

    d1 = (math.log(s / k) + (r + 0.5 * sig * sig) * t) / (sig * math.sqrt(t))
    d2 = d1 - sig * math.sqrt(t)

    if i.option_type == "call":
        return s * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(d2)
    if i.option_type == "put":
        return k * math.exp(-r * t) * _norm_cdf(-d2) - s * _norm_cdf(-d1)

    raise ValueError("option_type must be 'call' or 'put'")

