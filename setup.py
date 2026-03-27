"""setup.py — handles Cython extension build alongside pyproject.toml."""

from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    import numpy

    ext_period = Extension(
        "pycycle._ext._pycycle_c",
        sources=[
            "pycycle/_ext/_pycycle_c.pyx",
            "pycycle/_ext/pycycle_py_c.c",
        ],
        include_dirs=[numpy.get_include()],
    )
    ext_template = Extension(
        "pycycle._ext.template_fit_c",
        sources=["pycycle/_ext/template_fit_c.pyx"],
        include_dirs=[numpy.get_include()],
    )
    ext_modules = cythonize(
        [ext_period, ext_template],
        compiler_directives={"boundscheck": False, "wraparound": False},
    )
except Exception:
    ext_modules = []

setup(ext_modules=ext_modules)
