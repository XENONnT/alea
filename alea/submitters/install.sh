#!/bin/bash

set -e

# List of packages
packages=("alea")

# Loop through each package
for package in "${packages[@]}"
do
    # Check if the tarball exists
    if [ ! -f "$package.tar.gz" ]; then
        echo "Tarball $package.tar.gz not found. Skipping $package."
        echo
        continue
    fi

    echo "Installing $package:"

    # Create a directory for the package
    mkdir -p $package

    # Extract the tarball to the package directory
    tar -xzf $package.tar.gz -C $package --strip-components=1

    # Install the package in very quiet mode by -qq
    pip install ./$package --user --no-deps -qq

    # Verify the installation by importing the package
    python -c "import $package; print($package.__file__)"

    echo "$package installation complete."
    echo
done
