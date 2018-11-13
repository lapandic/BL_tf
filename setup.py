from setuptools import setup

setup(
    name='BL_tensorflow',
    version='0.0.1',
    keywords='duckietown, logs, imitation learning, tensorflow',
    install_requires=[
        'h5py', #>=2.8.0',
        'pyyaml',
        'rospkg',
        'sklearn',
        'scipy',
        'opencv-python',
        'numpy<=1.14.5',
        'pandas>=0.23.0',
        'tables>=3.4.3',
        'tensorboard>=1.8.0',
        'tensorflow>=1.8.0',
        'requests',
    ],
    entry_points={
        'console_scripts': [
            'BL_tensorflow-start=BL_tensorflow.launcher:main',
        ],
    },
)
