import autoreduce


def test_import():
    assert autoreduce.__version__


def test_subpackages_import():
    from autoreduce import acquire, align, drizzle, noise, psf, package, instruments

    for module in (acquire, align, drizzle, noise, psf, package, instruments):
        assert module.__doc__
