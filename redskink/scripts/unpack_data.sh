#!/bin/bash

OSG_path=$1

for file in `ls $OSG_path/*discovery*.tar.gz`
do
  tar xvfz $file -C $OSG_path
done

for file in `ls $OSG_path/*sensi*.tar.gz`
do
  tar xvfz $file -C $OSG_path
done
