from setuptools import setup, find_packages

setup(
    name="PlantReactivityAnalysis",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    description="The code explores the hypothesis that plants can detect and respond to human movements, especially\
          eurythmy gestures, using plant-based electrical signals. We develop various machine learning models to \
            interpret these signals, treating plants as biosensors for human motion.",
    author="Alvaro Francisco Gil",
    license="MIT",
)
