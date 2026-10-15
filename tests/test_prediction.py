import pytest

from src.recommendations import generate_recommendation


def test_high_risk_recommendation():
    result = generate_recommendation(0.85)

    assert result["risk_level"] == "HIGH"


def test_medium_risk_recommendation():
    result = generate_recommendation(0.55)

    assert result["risk_level"] == "MEDIUM"


def test_low_risk_recommendation():
    result = generate_recommendation(0.15)

    assert result["risk_level"] == "LOW"


def test_invalid_probability():
    with pytest.raises(ValueError):
        generate_recommendation(1.5)