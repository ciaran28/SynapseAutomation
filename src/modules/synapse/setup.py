import setuptools
from datetime import datetime
from pathlib import Path
import sys

def main(version):
    current_file = Path(__file__).absolute()
    print(f'Current file path: {current_file}')
    current_file_folder = Path(__file__).parent.absolute()
    print(f"Current folder: {current_file_folder}")

    package_name = 'dstoolkit'

    path_requirements = current_file_folder.joinpath('requirements.txt')

    with open(path_requirements, "r") as fh:
        requirements = [l.strip() for l in fh.readlines()]

    requirements = [rq for rq in requirements if (rq) and (rq.startswith('#') is False)]

    # today = datetime.today()
    # version = f'{today:%Y}{today:%m}{today:%d}_{today:%H}{today:%M}{today:%S}'

    setuptools.setup(
        name=package_name,
        version=version,
        author="Davide Fornelli",
        author_email="daforne@microsoft.com",
        description="Data Science Toolkit library to accelerate Azure Synapse development",
        packages=[package_name],
        install_requires=requirements
        # python_requires='~=3.8'
    )



if __name__ == "__main__":

    if "--version" in sys.argv:        
        dx = [i+1 for i,x in enumerate(sys.argv) if x == '--version'][0]
        version = sys.argv[dx]
        sys.argv.remove("--version")
        sys.argv.remove(version)
    else:
        version="1.0"
              
    main(version)