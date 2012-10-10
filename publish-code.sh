#!/bin/sh

TAG=$(date +"release-%Y-%m-%d")

git tag -a $TAG -m "$TAG"
git archive --prefix=mca-bsc-code/ HEAD | gzip > ~/public_html/NonLinSC/mca-bsc-code.tgz

