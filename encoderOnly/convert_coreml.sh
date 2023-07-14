#!/bin/bash
if [[ $# -gt 0 ]]
  then
      echo ""
  else
      echo please provide model size like tiny, small, large...
      exit
fi

python convert_encoder.py $1

# build shared library
echo ""
echo "--------------------------"
echo "🦙 Build Shared Library 🦙"
echo "--------------------------"
cd coreml
model=$1 make clean
model=$1 make

echo "---------------------"
echo "🦊 Run EncoderTest 🦊"
echo "---------------------"
$1/encoderTest
