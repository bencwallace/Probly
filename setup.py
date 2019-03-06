import pathlib
from setuptools import setup

here = pathlib.Path(__file__).parent
version = (here / 'version').read_text()
README = (here / 'README.md').read_text()


with open('README.md') as f:
    long_description = f.read()

setup(name='probly',
      version=version,
      description='Probabilistic computations in Python',
      long_description=README,
      long_description_content_type='text/markdown',
      url='https://probly.readthedocs.io/',
      author='Benjamin Wallace',
      author_email='bencwallace@gmail.com',
      classifiers=[
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Education',
                   'License :: OSI Approved :: BSD License',
                   'Topic :: Scientific/Engineering :: Mathematics'
      ],
      packages=['probly'],
      install_requires=['numpy'],
      include_package_data=True,
      zip_safe=False)
