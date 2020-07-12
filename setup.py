import pathlib
from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent
version = (here / 'version').read_text()
README = (here / 'README.rst').read_text()

setup(name='probly',
      version=version,
      description='A Python package for the symbolic computation of'
                  ' random variables.',
      long_description=README,
      long_description_content_type='text/markdown',
      url='https://probly.readthedocs.io/',
      author='Benjamin Wallace',
      author_email='bencwallace@gmail.com',
      classifiers=['Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Education',
                   'License :: OSI Approved :: BSD License',
                   'Topic :: Scientific/Engineering :: Mathematics'],
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'matplotlib'],
      include_package_data=True,
      zip_safe=False)
