from setuptools import setup
import os

HERE = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(HERE, "README.md")) as fid:
    README = fid.read()

setup(
    name='spela',
    version='0.0.4',
    packages=['spela', 'utils'],
    url='https://github.com/kongkip/spela.git',
    long_description=str(README),
    long_description_content_type="text/markdown",
    author='Evans Kiplagat',
    author_email='evanskiplagat3@gmail.com',
    license='MIT',
    description='spectrogram layers',
    install_requires=['numpy', 'librosa', 'matplotlib']
)
