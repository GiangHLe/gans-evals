import setuptools
import os

def read(rel_path):
    base_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_path, rel_path), 'r') as f:
        return f.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError('Unable to find version string.')
    
setuptools.setup(
            name='gans-eval',
            version=get_version(os.path.join('src', 'gans_eval', '__init__.py')),
            url='https://github.com/GiangHLe/gans-evals',
            author='Giang Le',
            author_email='lhgiang149@gmail.com',
            description=('This package provides measurement tools for Generative Adversarial Networks (GANs), including Inception Score (IS), Fréchet Inception Distance (FID), Kernel Inception Distance (KID), and Precision and Recall (PR). These metrics are used to evaluate the quality and diversity of generated images in GANs. The package streamlines the use of these metrics, making it easier to apply them to your work.'),
            long_description=read('README.md'),
            long_description_content_type='text/markdown',
            install_requires=['numpy',
                            'opencv-python',
                            'pillow',
                            'scipy',
                            'tqdm',
                            'timm',
                            'torch>=1.7.0',
                            'torchvision>=0.8.0'],

            classifiers=[
                'Programming Language :: Python :: 3',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: MIT License',  
                'Operating System :: POSIX :: Linux',
                'Programming Language :: Python :: 3',
            ],
            package_dir={"": "src"},
            packages = setuptools.find_packages(where="src"),
            python_requires = ">=3.6",
            entry_points={
            'console_scripts': [
                'gans-eval = gans_eval.main:main',
            ]})