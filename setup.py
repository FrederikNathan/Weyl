#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:11:46 2020

@author: frederik
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("time_domain_pyx.pyx")
)