# Licensed under a 3-clause BSD style license - see LICENSES

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from astropy import units as u

import sncosmo


def test_abmagsystem():
    magsys = sncosmo.ABMagSystem()
    m = magsys.band_flux_to_mag(1.0, 'bessellb')
    f = magsys.band_mag_to_flux(m, 'bessellb')
    assert_almost_equal(f, 1.0)


def test_spectralmagsystem():
    """Check that SpectralMagSystem matches ABMagSystem when the spectrum is
    the same as AB."""

    # construct a spectrum with same flux as AB: 3631 x 10^{-23} erg/s/cm^2/Hz
    wave = np.linspace(1000., 20000., 1000)
    flux = 3631.e-23 * np.ones_like(wave)
    unit = u.erg / u.s / u.cm**2 / u.Hz
    s = sncosmo.Spectrum(wave, flux, unit=unit)
    magsys1 = sncosmo.SpectralMagSystem(s)

    magsys2 = sncosmo.ABMagSystem()

    # Match is only good to 1e-3. Probably due to different methods
    # of calculating bandflux between Spectrum and ABMagSystem.
    assert_allclose(magsys1.zpbandflux('bessellb'),
                    magsys2.zpbandflux('bessellb'), rtol=1e-3)


# issue 100
def test_bandpass_type():
    """Check that bandpass wavelength type is always float64,
    and that color laws work with them."""

    dust = sncosmo.CCM89Dust()

    for dt in [np.int32, np.int64, np.float32]:
        wave = np.arange(4000., 5000., 20., dtype=dt)
        trans = np.ones_like(wave)
        band = sncosmo.Bandpass(wave, trans)

        assert band.wave.dtype == np.float64

        # Ensure that it works with cython-based propagation effect.
        dust.propagate(band.wave, np.ones_like(wave))


# issue 111
def test_bandpass_bessell():
    """Check that Bessell bandpass definitions are scaled by inverse
    wavelength."""

    band = sncosmo.get_bandpass('bessellb')
    trans = band.trans[[4, 9, 14]]  # transmission at 4000, 4500, 5000

    # copied from file
    orig_wave = np.array([4000., 4500., 5000.])
    orig_trans = np.array([0.920, 0.853, 0.325])

    scaled_trans = orig_trans / orig_wave

    # scaled_trans should be proportional to trans
    factor = scaled_trans[0] / trans[0]
    assert_allclose(scaled_trans, factor * trans)
