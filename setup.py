"""setup.py — handles Cython extension build alongside pyproject.toml."""

from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    import numpy

    ext = Extension(
        "psearch._ext._psearch_c",
        sources=[
            "psearch/_ext/_psearch_c.pyx",
            "psearch/_ext/psearch_py_c.c",
        ],
        include_dirs=[numpy.get_include()],
    )
    ext_modules = cythonize(
        [ext],
        compiler_directives={"boundscheck": False, "wraparound": False},
    )
except Exception:
    ext_modules = []

setup(ext_modules=ext_modules)
