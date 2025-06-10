from setuptools import setup

setup(
    name='custom_envs', #同層のディレクトリ名
    version='0.0.2',
    install_requires=[
    'gymnasium>=1.0.0',
    'numpy>=2.0.0',
    ],
    packages=['custom_envs', 'custom_envs.envs'],
)