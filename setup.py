from setuptools import setup, find_packages
setup(
    name = "trajnetbaselines",
    #packages=find_packages(),
    packages=[
        'trajnetbaselines',
        'trajnetbaselines.lstm',
        'random_smooth',
        'evaluator',
<<<<<<< Updated upstream
=======
<<<<<<< Updated upstream
        'bounded_regression'
=======
>>>>>>> Stashed changes
        'bounded_regression',
        'diffusion_bound_regression',
        'diffusion_bound_regression.MID_from_git.environment',
        'diffusion_bound_regression.MID_from_git.models',
<<<<<<< Updated upstream
        'diffusion_bound_regression.MID_from_git.utils',
=======
        'diffusion_bound_regression.MID_from_git.models.encoders.components',
        'diffusion_bound_regression.MID_from_git.utils',
>>>>>>> Stashed changes
>>>>>>> Stashed changes
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