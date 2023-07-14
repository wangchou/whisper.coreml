### Encoder only conversion script
```sh
# converting large model will take about 30s
# it runs encoderTest at the end (currently it is hard-coded to small model)
cd encoderOnly
./convert_coreml.sh [tiny|...|...]
```

### Test it on audio file
```sh
# 1. convert encoder, decoder and build shared library
# for small model will take about 70s (3 coreml models)
./convert_coreml.sh [tiny|small|medium|...]

# 2. run transcribe
python -m whisper YOUR_WAV_FILE --language=[ja|en|...] --model=$1 --beam_size=5 --best_of=5 --word_timestamps=True --use_coreml=True

# Known constraints:
# 1. --beam_size and --best_of is fixed to 5
# 2. specifying --language is required

# Known issues:
# 1. Large model won't work (Decoder256 crashes)
# 2. ANECompilerServices takes a long time on medium and large decoder model
```

