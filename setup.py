from setuptools import setup, find_packages
setup(
    name = "trajnetbaselines",
    #packages=find_packages(),
    packages=[
        'trajnetbaselines',
        'trajnetbaselines.lstm',
    ],
    install_requires=[
        'numpy',
        'pykalman',
        'python-json-logger',
        'scipy',
        'torch',
        'trajnetplusplustools',
        'pysparkling',
        'joblib',
        'pandas',
        'matplotlib',
        'torchvision',
        'tqdm',
    ],
)