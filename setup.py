from setuptools import setup, find_packages
import codecs
import re
import os

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# read the contents of the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='das',
      version=find_version("src/das/__init__.py"),
      description='deep audio segmenter',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/janclemenslab/das',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=['numpy', 'h5py', 'scipy', 'scikit-klearn', 'pyyaml', 'peakutils', 'zarr',
                        'flammkuchen', 'defopt', 'matplotlib', 'pandas', 'librosa', 'matplotlib',
                        'matplotlib_scalebar', 'peakutils'],
      include_package_data=True,
      zip_safe=False,
      entry_points = {'console_scripts': ['das=das.cli:main']}
     )

print('IMPORTANT:')
print('Tensorflow is required to run DeepSS but is not installed automatically,')
print('to avoid interference with existing installations.')
print('')
print('Run `pip install tensorflow` to install tensorflow.')
