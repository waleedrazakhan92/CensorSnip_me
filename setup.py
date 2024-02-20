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
            'dummy_main = dummy_main:main',
        ],
    },
    author='Waleed Raza',
    author_email='waleedrazakhan92@gmail.com',
    description='A package for performing inference using YOLOv8 model.',
    url='https://github.com/techtative/CensorSnip',
)
