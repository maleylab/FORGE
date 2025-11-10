from setuptools import setup, find_packages
import pathlib, os

# Detect layout
use_src = pathlib.Path("src/labtools").exists()
pkg_args = {"package_dir": {"": "src"}, "packages": find_packages(where="src")} if use_src \
           else {"packages": find_packages(where=".")}

setup(
    name="lab-tools",
    version="0.1.0",
    include_package_data=True,
    **pkg_args
)
