from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['gym_fbot'],
    package_dir={'': 'src'}
)

setup(**d)