import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()


with open('README.md') as f:
    long_description = f.read()

setup(name='probly',
      version='0.1.0',
      description='Probabilistic computations in Python',
      long_description=README,
      long_description_content_type='text/markdown',
      url='https://github.com/bencwallace/probly',
      author='Benjamin Wallace',
      author_email='bencwallace@gmail.com',
      classifiers=[
                   'Development Status :: 2 - Pre-Alpha',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Education',
                   'License :: OSI Approved :: BSD License',
                   'Topic :: Scientific/Engineering :: Mathematics'
      ],
      packages=['probly'],
      install_requires=['numpy', 'networkx'])
