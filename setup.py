#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Dedi Wang",
    author_email='wangdedi1997@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This is a WESTPA 2.0 plug-in for SPIB augmented weighted ensemble.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='spib_we',
    name='spib_we',
    packages=find_packages(include=['spib_we', 'spib_we.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/wangdedi1997/spib_we',
    version='0.1.0',
    zip_safe=False,
)
