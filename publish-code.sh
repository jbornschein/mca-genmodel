#!/bin/sh

git archive --prefix=mca-bsc-code/ HEAD | gzip > ~/public_html/NonLinSC/mca-bsc-code.tgz

