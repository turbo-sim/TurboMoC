"""Flow-domain schema migration preserves numerical arguments and geometry."""
from copy import deepcopy

import numpy as np
import pytest

from turbo_moc.configuration.flow_domain import normalize, to_flat, to_passage, DEFAULTS
from turbo_moc.configuration import project as projects
from turbo_moc.geometry import blade_row
from turbo_moc.geometry.rotor_blade import _flow_domain


LEGACY = {
    "domain_shift_mode": "manual", "domain_shift": .07, "num_points": 43,
    "inlet_chords": .75, "outlet_chords": 1.25,
    "inlet_extension_type": "straight", "outlet_extension_type": "bspline",
    "inlet_axial_tangent": .21, "inlet_blade_tangent": .32,
    "outlet_axial_tangent": .43, "outlet_blade_tangent": .24,
}


@pytest.mark.parametrize("smooth", [False, True])
def test_legacy_domain_migration_preserves_every_value_and_is_idempotent(smooth):
    legacy = deepcopy(LEGACY)
    if smooth:
        fields = {key for key in legacy if 'tangent' in key or 'type' in key or key == 'num_points'}
        legacy['smooth'] = {key: legacy.pop(key) for key in fields}
    original = deepcopy(legacy)
    nested = normalize(legacy)
    assert list(nested) == ['domain_shift_mode', 'domain_shift_value', 'number_of_points', 'inlet', 'outlet']
    assert nested['number_of_points'] == 43
    assert nested['inlet']['extension_length'] == .75
    assert to_flat(nested) == LEGACY
    assert normalize(nested) == nested
    assert legacy == original


def test_partial_rotor_domain_retains_saved_values_when_defaults_are_added():
    nested = normalize({'inlet_chords': .9, 'smooth': {'num_points': 47}}, defaults=DEFAULTS)
    assert nested['inlet']['extension_length'] == .9
    assert nested['number_of_points'] == 47
    assert nested['outlet'] == DEFAULTS['outlet']


@pytest.mark.parametrize('value', [None, {'inlet': 1}, {'outlet': {'typo': 2}},
    {'domain_shift': .1, 'domain_shift_value': .2},
    {'inlet_chords': .3, 'inlet': {'extension_length': .4}}])
def test_malformed_or_ambiguous_domain_is_rejected(value):
    with pytest.raises(ValueError, match='flow_domain'):
        normalize(value)


@pytest.mark.parametrize('family', ['linear', 'axial', 'radial'])
def test_nested_and_legacy_domains_produce_identical_stator_geometry(small_design, family):
    nozzle, config = small_design
    config['meridional_channel'] = projects.row_configuration(projects.default_project(family), 'stator_1')['meridional_channel']
    config['meridional_channel']['num_blades'] = 12
    if family == 'linear':
        config['meridional_channel'].pop('num_blades')
    config['flow_domain'] = normalize(LEGACY)
    # Automatic placement avoids a deliberately offset passage intersecting the blade.
    config['flow_domain']['domain_shift_mode'] = 'auto'
    old = deepcopy(config)
    old['flow_domain'] = to_flat(config['flow_domain'])
    current = blade_row.build_design(nozzle, config)
    previous = blade_row.build_design(nozzle, old)
    for name in ('inlet', 'outlet', 'periodic_left', 'periodic_right', 'blade_curve'):
        np.testing.assert_array_equal(current.curves[name], previous.curves[name])


def test_nested_and_legacy_domains_produce_identical_rotor_extensions():
    # A simple closed blade isolates the rotor passage adapter from thermodynamics.
    blade = {'blade_curve': {'x': [0., .1, .1, 0., -.1, -.1, 0.],
                             'y': [0., -.3, -.8, -1., -.8, -.3, 0.]},
             'trailing_edge': {'x': [], 'y': []}, 'pitch': 2., 'metal_angle_in': 20., 'metal_angle_out': -20.}
    nested, flat = _flow_domain(blade, normalize(LEGACY)), _flow_domain(blade, LEGACY)
    for name in ('inlet_extension', 'outlet_extension', 'periodic_left', 'periodic_right'):
        np.testing.assert_array_equal(nested[name], flat[name])
    assert nested['domain_shift'] == flat['domain_shift'] == LEGACY['domain_shift']


def test_planar_adapter_preserves_legacy_smooth_contract():
    passage = to_passage(normalize(LEGACY))
    assert passage['inlet_chords'] == LEGACY['inlet_chords']
    assert passage['domain_shift'] == LEGACY['domain_shift']
    assert passage['smooth']['num_points'] == LEGACY['num_points']
    assert normalize(passage) == normalize(LEGACY)
