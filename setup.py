#!/usr/bin/env python
from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as req_fp:
    install_requires = req_fp.readlines()

setup(
    name='wallabynmt',
    version='1.2',
    description='JoeyNMT with continuous outputs',
    author='Samuel Kiegeland and Marvin Koss (JoeyNMT by Jasmijn Bastings and Julia Kreutzer)',
    url='https://github.com/continue-us/wallabynmt',
    license='Apache License',
    install_requires=install_requires,
    packages=find_packages(exclude=[]),
    python_requires='>=3.5',
    project_urls={
        'Documentation': 'http://joeynmt.readthedocs.io/en/latest/',
        'Source': 'https://github.com/joeynmt/joeynmt',
        'Tracker': 'https://github.com/joeynmt/joeynmt/issues',
    },
    entry_points={
        'console_scripts': [
        ],
    }
)
