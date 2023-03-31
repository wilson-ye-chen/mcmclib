from setuptools import setup

setup(
    name='mcmclib',
    version='0.1.0',
    description='Library of MCMC sampling algorithms',
    url='https://github.com/wilson-ye-chen/mcmclib',
    author='Wilson Ye Chen',
    license='MIT',
    packages=['mcmclib'],
    install_requires=['numpy', 'scipy', 'tqdm']
    )
