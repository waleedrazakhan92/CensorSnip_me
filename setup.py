from setuptools import setup, find_packages

setup(
    name='CensorSnip',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'ffmpegcv',
        'roboflow',
        'ipython',
        'matplotlib',
        'moviepy',
        'tqdm',
        'ultralytics'
    ],
    entry_points={
        'console_scripts': [
            'run_inference=CensorSnip_pkg.run_inference:run',
        ],
    },

)
