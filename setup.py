from setuptools import setup, find_packages
setup(
    name = "trajnetbaselines",
    #packages=find_packages(),
    packages=[
        'trajnetbaselines',
        'trajnetbaselines.lstm',
        'random_smooth',
        'evaluator',
        'bounded_regression',
        'diffusion_bound_regression',
        'diffusion_bound_regression.MID_from_git.environment',
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