from pathlib import Path
from setuptools import find_packages, setup


DESCRIPTION = 'LightOn technologies for Large scale Machine Learning with Optical Processing Unit'
__here__ = Path(__file__).absolute().parent

setup(
    name='lightonml',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author='LightOn AI Research',
    author_email='iacopo@lighton.io,charles@lighton.io,igor@lighton.io',
    url="https://docs.lighton.ai",
    description=DESCRIPTION,
    long_description=(__here__/"public"/"README.md").read_text(),
    long_description_content_type='text/markdown',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License"
    ],
    entry_points={
        # opu_test as console script
        "console_scripts": ["opu_test=lightonml.cmd_line.opu_test:main"]
    },
    packages=find_packages(exclude=['test', 'docs', 'test*', 'examples']),
    # add extension module
    install_requires=["numexpr==2.7.*", "numpy", "requests", "attrs>=19"],
    extras_require={"torch": ["torch>=1.0"],
                    "sklearn": ["scikit-learn"],
                    "drivers": ["lightonopu>=1.4.2"]},
    zip_safe=False
)
