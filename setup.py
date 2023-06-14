"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

from setuptools import setup

setup(
    name="CompanyKG",
    version="1.0",
    package_dir={"": "src"},
    include_package_data=True,
    description="Company Knowledge Graph data loading and evaluation utilities",
    author="EQT Motherbrain",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "scikit-learn",
        "pyarrow",
        "fastparquet",
    ],
)
