# Whisper+Coreml
Whisper+Coreml speeds up decoder and encoder by using Apple Neural Engine (ANE).

### Usage
```sh
# 1. convert encoder, decoder to coreml model and build shared library
#    (small model: 70s, large model: 5mins)
./convert_coreml.sh [tiny|small|...|turbo] [beam_size]

# 2. transcribe
python -m whisper YOUR_WAV_FILE --language=[ja|en|...] --model=$1 --beam_size=beam_size --best_of=beam_size --word_timestamps=True --use_coreml=True

# Known constraints:
# 1. beam_size and best_of are fixed on each coreml model
# 2. specifying --language is required
```

### Performance
* transcribe() 1 mins song on Macbook M1 Air 16GB with **beam_size=1**

|  Model Size  | 1st load time | cached load time | transcribe time (bs=1)|
|:------:|----------:|------------------:|------------------:|
| turbo (openai/whisper cpu)  |     |            |      21s       |

* transcribe() 1 mins song on Macbook M1 Air 16GB with **beam_size=5** (default option of openai/whisper)

|  Model Size  | 1st load time | cached load time | transcribe time (bs=5)|
|:------:|----------:|------------------:|------------------:|
| turbo (openai/whisper cpu)  |     |            |      32s       |
| turbo (whisper+coreml **default**) |  4s   |    1.5s        |      load time + 9.4s       |
| turbo (whisper+coreml **encoder on ane**)  |  4m14s   |    1.5s        |      load time + 7.0s       |


**Note**:

* Transcribe time means the time of transcribe() in transcribe.py. Python model load time is not included.
* turbo model default: 
  * encoder on GPU
  * crossKVCaches on ANE
  * decoder 256 on ANE
  * decoder1 on GPU
* turbo model with **encoder on ANE**: encoder speed 3x faster but with 4min initial load time penality. (edit coreml/coreml.mm to switch GPU/ANE mode)

<img src="./img/first_load_time.jpg" alt="isolated" width="400"/>
