from setuptools import setup, find_packages

setup(
    name = "project47",
    version = '0.0.0',
    packages = find_packages(),
    install_requires=[
        'utm',
        'numpy',
        'matplotlib',
        'cartopy',
        'ortools',
        'osmnx',
        'shapely',
        'pandas',
        'googlemaps',
        'geopy'

    ]
    
)