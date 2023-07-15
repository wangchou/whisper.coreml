#usage: model=[tiny|base|small|medium|large] make [all|clean]
MODEL = $$model

all: $(MODEL)/encoderTest $(MODEL)/decoderTest $(MODEL)/decoder256Test $(MODEL)/crossKVTest

$(MODEL)/encoderTest: encoderTest.cpp $(MODEL)/encoderWrapper.so
	cp encoderTest.cpp ${@D}
	c++ -I${@D} ${@D}/encoderTest.cpp ${@D}/encoderWrapper.so -o ${@D}/encoderTest

$(MODEL)/encoderWrapper.so: encoderWrapper.h encoderWrapper.mm
	cp encoderWrapper* ${@D}
	cp coremlUtility* ${@D}
	xcrun coremlc generate  ${@D}/CoremlEncoder0.mlpackage ${@D}
	for mlmodel in $(shell find $(MODEL) -name 'CoremlEncoder*.mlpackage') ; do \
		xcrun coremlc compile  $$mlmodel ${@D} || exit 1; \
    done
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/encoderWrapper.mm ${@D}/coremlUtility.mm ${@D}/CoremlEncoder0.m -o ${@D}/encoderWrapper.so -framework Foundation -framework CoreML -framework Accelerate

$(MODEL)/decoderTest: decoderTest.cpp $(MODEL)/decoderWrapper.so
	cp decoderTest.cpp ${@D}
	c++ -I${@D} ${@D}/decoderTest.cpp ${@D}/decoderWrapper.so -o ${@D}/decoderTest

$(MODEL)/decoderWrapper.so: decoderWrapper.h decoderWrapper.mm
	cp decoderWrapper* ${@D}
	cp coremlUtility* ${@D}
	xcrun coremlc compile ${@D}/CoremlDecoder.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/decoderWrapper.mm ${@D}/CoremlDecoder.m ${@D}/coremlUtility.mm -o ${@D}/decoderWrapper.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

$(MODEL)/decoder256Test: decoder256Test.cpp $(MODEL)/decoder256Wrapper.so
	cp decoder256Test.cpp ${@D}
	c++ -I${@D} ${@D}/decoder256Test.cpp ${@D}/decoder256Wrapper.so -o ${@D}/decoder256Test

$(MODEL)/decoder256Wrapper.so: decoder256Wrapper.h decoder256Wrapper.mm
	cp decoder256Wrapper* ${@D}
	cp coremlUtility* ${@D}
	xcrun coremlc compile ${@D}/CoremlDecoder256.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder256.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/decoder256Wrapper.mm ${@D}/CoremlDecoder256.m ${@D}/coremlUtility.mm -o ${@D}/decoder256Wrapper.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

$(MODEL)/crossKVTest: crossKVTest.cpp $(MODEL)/crossKVWrapper.so
	cp crossKVTest.cpp ${@D}
	c++ -I${@D} ${@D}/crossKVTest.cpp ${@D}/crossKVWrapper.so -o ${@D}/crossKVTest

$(MODEL)/crossKVWrapper.so: crossKVWrapper.h crossKVWrapper.mm
	cp crossKVWrapper* ${@D}
	cp coremlUtility* ${@D}
	xcrun coremlc compile ${@D}/CoremlCrossKV.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlCrossKV.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/crossKVWrapper.mm ${@D}/CoremlCrossKV.m ${@D}/coremlUtility.mm -o ${@D}/crossKVWrapper.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

clean:
	rm -rf $(MODEL)/*.mlmodelc
	rm -f $(MODEL)/*.h
	rm -f $(MODEL)/*.m
	rm -f $(MODEL)/*.mm
	rm -f $(MODEL)/*.cpp
	rm -f $(MODEL)/*Test
	rm -f $(MODEL)/*.so
