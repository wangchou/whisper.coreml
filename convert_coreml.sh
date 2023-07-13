#!/bin/bash
if [[ $# -gt 0 ]]
  then
      echo ""
  else
      echo please provide model size like tiny, small, large...
      exit
fi

python convert_encoder.py $1
python convert_decoder.py $1
python convert_decoder256.py $1

# build shared library
cd coreml
model=$1 make clean
model=$1 make

echo ""
echo "----"
echo "usage: python -m whisper YOUR_WAV_FILE --language=[ja|en|...] --model=$1 --beam_size=5 --best_of=5 --word_timestamps=True --use_coreml=True"
