#usage: model=[tiny|base|small|medium|large] make [all|clean]
#MODEL = $$model

all: small/encoderTest small/decoderTest small/decoder256Test small/crossKVTest

small/encoderTest: encoderTest.cpp small/encoder.so
	cp encoderTest.cpp ${@D}
	c++ -I${@D} ${@D}/encoderTest.cpp ${@D}/encoder.so -o ${@D}/encoderTest

small/encoder.so: encoder.h encoder.mm
	cp encoder* ${@D}
	cp coremlUtility* ${@D}
	xcrun coremlc generate  ${@D}/CoremlEncoder0.mlpackage ${@D}
	for mlmodel in $(shell find small -name 'CoremlEncoder*.mlpackage') ; do \
		xcrun coremlc compile $$mlmodel ${@D} || exit 1; \
    done
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/encoder.mm ${@D}/coremlUtility.mm ${@D}/CoremlEncoder0.m -o ${@D}/encoder.so -framework Foundation -framework CoreML -framework Accelerate

small/decoderTest: decoderTest.cpp small/decoder.so
	cp decoderTest.cpp ${@D}
	c++ -I${@D} ${@D}/decoderTest.cpp ${@D}/decoder.so -o ${@D}/decoderTest

small/decoder.so: decoder.h decoder.mm
	cp decoder* ${@D}
	cp coremlUtility* ${@D}
	xcrun coremlc compile ${@D}/CoremlDecoder.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/decoder.mm ${@D}/CoremlDecoder.m ${@D}/coremlUtility.mm -o ${@D}/decoder.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

small/decoder256Test: decoder256Test.cpp small/decoder256.so
	cp decoder256Test.cpp ${@D}
	c++ -I${@D} ${@D}/decoder256Test.cpp ${@D}/decoder256.so -o ${@D}/decoder256Test

small/decoder256.so: decoder256.h decoder256.mm
	cp decoder256* ${@D}
	cp coremlUtility* ${@D}
	xcrun coremlc compile ${@D}/CoremlDecoder256.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder256.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/decoder256.mm ${@D}/CoremlDecoder256.m ${@D}/coremlUtility.mm -o ${@D}/decoder256.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

small/crossKVTest: crossKVTest.cpp small/crossKV.so
	cp crossKVTest.cpp ${@D}
	c++ -I${@D} ${@D}/crossKVTest.cpp ${@D}/crossKV.so -o ${@D}/crossKVTest

small/crossKV.so: crossKV.h crossKV.mm
	cp crossKV* ${@D}
	cp coremlUtility* ${@D}
	xcrun coremlc compile ${@D}/CoremlCrossKV.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlCrossKV.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/crossKV.mm ${@D}/CoremlCrossKV.m ${@D}/coremlUtility.mm -o ${@D}/crossKV.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

clean:
	rm -rf small/*.mlmodelc
	rm -f small/*.h
	rm -f small/*.m
	rm -f small/*.mm
	rm -f small/*.cpp
	rm -f small/*Test
	rm -f small/*.so
