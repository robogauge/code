from setuptools import setup, find_packages

setup(
    name="robogauge",
    version="1.1.1",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="A generic robot RL model evaluation library based on MuJoCo",
    packages=find_packages(),
    install_requires=[
        "torch",  # Refer: https://pytorch.org/get-started/locally/
        "numpy==1.20.0",
        "pillow==9.0.0",
        "mujoco>=3.0.0",
        "dm_control>=1.0.14",
        "scipy",
        "matplotlib==3.6.3",
        "tqdm",
        "imageio[ffmpeg]",
        "tensorboard",
        "PyYAML",
        "fastapi",
        "uvicorn",
    ],
    python_requires=">=3.8",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
)
