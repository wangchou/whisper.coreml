# Whisper+Coreml
Whisper+Coreml speeds up decoder and encoder by using Apple Neural Engine (ANE).

### Usage
```sh
# 1. convert encoder, decoder to coreml model and build shared library
#    (small model: 70s, large model: 5mins)
./convert_coreml.sh [tiny|small|medium|...] [beam_size]

# 2. transcribe
python -m whisper YOUR_WAV_FILE --language=[ja|en|...] --model=$1 --beam_size=beam_size --best_of=beam_size --word_timestamps=True --use_coreml=True

# Known constraints:
# 1. beam_size and best_of are fixed on each coreml model
# 2. specifying --language is required
```

### Performance
* transcribe() 1 mins song on Macbook M1 Air 16GB when **beam_size=1**

|  Model Size  | 1st load time | cached load time | transcribe time (bs=1)|
|:------:|----------:|------------------:|------------------:|
| small (whisper+coreml ane)  |   47s    |     1s     |      load time + 2s |
| small (openai/whisper cpu)  |       |         |   9s   |
| large (whisper+coreml ane)  |   3m20s   |    9s        |      load time + 10s       |
| large (openai/whisper cpu)  |     |           |     42s  |

* transcribe() 1 mins song on Macbook M1 Air 16GB when **beam_size=5** (default option of openai/whisper)

|  Model Size  | 1st load time | cached load time | transcribe time (bs=5)|
|:------:|----------:|------------------:|------------------:|
| small (whisper+coreml ane)  |   55s    |     1s     |      load time + 4s |
| small (openai/whisper cpu)  |       |         |   27s   |
| large (whisper+coreml ane)  |   3m57s   |    10s        |      load time + 23s       |
| large (openai/whisper cpu)  |     |           |     122s  |

**Note**: Transcribe time only measure the time of transcribe() in transcribe.py. Python model load time is not included in transcribe time.

### Known issues:
* Loading coreml model for first time takes a long time on ANECompilerService (small model:50s, large model: 3m20s)
* Decoder256 of large model runs on GPU (memory issue? on M1 Air 16GB)

