from setuptools import setup

REQUIRED_PACKAGES = [
    'geopy==1.19.0', 'networkx==2.3',
    'scipy==1.2.1', 'pandas==0.24.2'
]

setup(
    name='kharita',
    version='0.2',
    packages=['kharita'],
    install_requires=REQUIRED_PACKAGES,
    url='',
    license='',
    author='',
    author_email='',
    description=''
)

