"""
setup.py

Copyright (C) 2017  Patrick Schwab, ETH Zurich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from distutils.core import setup

setup(
    name='dream_parkinsons',
    version='1.0.0',
    packages=['dream_parkinsons', 'dream_parkinsons.apps', 'dream_parkinsons.data_access', 'dream_parkinsons.models'],
    url='www.mhsl.hest.ethz.ch',
    author='Patrick Schwab',
    author_email='patrick.schwab@hest.ethz.ch',
    license=open('LICENSE.txt').read(),
    long_description=open('README.txt').read(),
    install_requires=[
        "synapseclient >= 1.7.2",
        "numpy >= 1.13.1",
        "Keras >= 1.2.2",
        "scikit-learn == 0.19.0",
        "h5py == 2.7.0"
    ]
)
