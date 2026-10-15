import pandas as pd
import pytest

from src.data_validation import validate_data


def test_valid_dataset():
    df = pd.DataFrame({
        "Late_delivery_risk": [0, 1, 0, 1],
        "Sales": [100, 200, 150, 250],
    })

    assert validate_data(df) is True


def test_empty_dataset():
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        validate_data(df)


def test_missing_target():
    df = pd.DataFrame({
        "Sales": [100, 200, 150],
    })

    with pytest.raises(ValueError):
        validate_data(df)


def test_missing_target_values():
    df = pd.DataFrame({
        "Late_delivery_risk": [0, None, 1],
    })

    with pytest.raises(ValueError):
        validate_data(df)


def test_invalid_target_values():
    df = pd.DataFrame({
        "Late_delivery_risk": [0, 1, 2],
    })

    with pytest.raises(ValueError):
        validate_data(df)