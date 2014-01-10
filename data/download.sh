#!/bin/sh

BASE_URL="http://fias.uni-frankfurt.de/~bornschein/NonLinSC/data/"

PATCHES20_FILE="patches-20.h5"
PATCHES26_FILE="patches-26.h5"
PATCHES20_DOG_FILE="patches-20-dog.h5"
PATCHES26_DOG_FILE="patches-26-dog.h5"

PATCHES20_URL="$BASE_URL/$PATCHES20_FILE"
PATCHES26_URL="$BASE_URL/$PATCHES26_FILE"
PATCHES20_DOG_URL="$BASE_URL/$PATCHES20_DOG_FILE"
PATCHES26_DOG_URL="$BASE_URL/$PATCHES26_DOG_FILE"

DOWN_CMD="wget -c "


echo
echo "This script will download and unpack about 30GB training data. "
echo 
echo "Make sure you have wget installed."
echo 
echo 
echo "Press CTRL-C to abort...   (sleeping for 10 sec.)"

sleep 3 
echo -e "Allright, lets continue...\n"


echo "Downloading 20x20 patches from $PATCHES20_URL..."
$DOWN_CMD $PATCHES20_URL

echo "Downloading 26x26 patches from $PATCHES26_URL..."
$DOWN_CMD $PATCHES26_URL

echo "Downloading 20x20 DoG whitened patches from $PATCHES20_DOG_URL..."
$DOWN_CMD $PATCHES20_DOG_URL

echo "Downloading 26x26 DoG whitened patches from $PATCHES26_DOG_URL..."
$DOWN_CMD $PATCHES26_DOG_URL

