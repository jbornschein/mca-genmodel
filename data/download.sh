#!/bin/sh

BASE_URL="http://fias.uni-frankfurt.de/~bornschein/data/mca-et-package"

IMAGES_FILE="vanhateren-linear.h5.bz2"
PATCHES20_FILE="patches20.h5.bz2"
PATCHES26_FILE="patches26.h5.bz2"

IMAGES_URL="$BASE_URL/$IMAGES_FILE"
PATCHES20_URL="$BASE_URL/$PATCHES20_FILE"
PATCHES26_URL="$BASE_URL/$PATCHES26_FILE"

echo
echo "This script will download and unpack XXX GB training data. The resulting"
echo "files will occupy YYY GB of diskspace."
echo 
echo "Make sure you have wget and bunzip2 installed."
echo 
echo 
echo "Press CTRL-C to abort...   (sleeping for 10 sec.)"

sleep 10
echo -e "Allright, lets continue...\n"


echo "\n\nDownloading Van Hateren images from $IMAGES_URL...\n\n"
wget $IMAGES_URL
bunzip $IMAGES_FILE

echo "Downloading 20x20 pseudo-whitened patches from $PATCHES20_URL..."
wget $PATCHES20_URL
bunzip $PATCHES20_FILE

echo "Downloading 26x26 pseudo-whitened patches from $PATCHES26_URL..."
wget $PATCHES26_URL
bunzip $PATCHES26_FILE

