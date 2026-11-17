import numpy as np
import pytest

from turbo_moc.geometry.karhunen_loeve import SigmaProfile


def test_sigma_profile_smoothly_transitions_between_endpoint_values():
    profile = SigmaProfile(0.25, 0.75, sigma0=2.0, sigma1=1.0)
    samples = np.asarray([profile.sigma(value) for value in np.linspace(0.25, 0.75, 101)])
    assert samples[0] == pytest.approx(1.0)
    assert samples[-1] == pytest.approx(1.0)
    assert samples[50] == pytest.approx(2.0)
    assert np.all(np.diff(samples[:51]) >= 0.0)
    assert np.all(np.diff(samples[50:]) <= 0.0)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_sigma_profile_rejects_out_of_range_positions(value):
    profile = SigmaProfile(0.25, 0.75, sigma0=2.0, sigma1=1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        profile.sigma(value)


def test_sigma_profile_rejects_invalid_transition_endpoints():
    with pytest.raises(ValueError, match="endpoints"):
        SigmaProfile(0.75, 0.25, sigma0=2.0, sigma1=1.0)
