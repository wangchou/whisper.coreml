#!/bin/bash
if [[ $# -gt 1 ]]
  then
      echo ""
  else
      echo please provide model size and beam_size
      echo example: ./convert_coreml.sh small 1
      exit
fi

python convert_encoder.py $1
python convert_decoder.py $1 $2
python convert_decoder256.py $1
python convert_ckv.py $1

# build shared library
echo ""
echo "--------------------------"
echo "🦙 Build Shared Library 🦙"
echo "--------------------------"
cd coreml
model=$1 make clean
model=$1 make

echo ""
echo "-----------"
echo "🦊 Usage 🦊"
echo "-----------"

# Known constraints:
# 1. fixed beam_size
# 2. specifying --language is required
# 3. make sure sample_len < (256-3) when using whisper.DecodingOptions (whisper default is 224)
# 4. large decoder256 runs on GPU because of memory issue (not sure)

echo "python -m whisper YOUR_WAV_FILE --language=[ja|en|...] --model=$1 --beam_size=$2 --word_timestamps=True --use_coreml=True"
