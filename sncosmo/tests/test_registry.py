# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test registry functions."""

import numpy as np
import sncosmo


def test_register():
    disp = np.array([4000., 4200., 4400., 4600., 4800., 5000.])
    trans = np.array([0., 1., 1., 1., 1., 0.])

    # create a band, register it, make sure we can get it back.
    band = sncosmo.Bandpass(disp, trans, name='tophatg')
    sncosmo.register(band)
    assert sncosmo.get_bandpass('tophatg') is band

    # test deprecated path to registry
    band = sncosmo.Bandpass(disp, trans, name='tophatg2')
    sncosmo.registry.register(band)
    assert sncosmo.get_bandpass('tophatg2') is band


def test_retrieve_cases():
    for name in ['ab', 'Ab', 'AB']:  # Should work regardless of case.
        sncosmo.get_magsystem(name)
