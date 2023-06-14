"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

from setuptools import setup

setup(
    name="companykg-benchmarks",
    version="1.0",
    package_dir={"": "src"},
    include_package_data=True,
    description="Company Knowledge Graph benchmarking utilities",
    author="EQT Motherbrain",
    install_requires=[
        "companykg",
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "dgl",
        "scikit-learn",
        "igraph",
        "click",
        "tqdm",
        "torch",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "torch-geometric",
        "deepsnap",
        "DGL",
        "PyGCL @ git+https://github.com/ivanustyuzhEQT/PyGCL",
        "numba",
    ],
    dependency_links=["https://data.dgl.ai/wheels/repo.html"],
)
