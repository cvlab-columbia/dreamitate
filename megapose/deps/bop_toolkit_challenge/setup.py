from setuptools import setup, find_packages

setup(
    name='bop_toolkit_lib',
    version='1.0',
    packages=find_packages(exclude=('docs')),
    install_requires=["pytz"],
    author='Tomas Hodan, Martin Sundermeyer',
    author_email='tom.hodan@gmail.com, Martin.Sundermeyer@dlr.de',
    license='MIT license',
    package_data={'bop_toolkit_lib':['*']},
)