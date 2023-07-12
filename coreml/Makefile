small: small/encoderWrapper.so small/decoderTest small/decoder256Test

small/encoderWrapper.so: encoderWrapper.h encoderWrapper.mm
	cp encoderWrapper* ${@D}
	xcrun coremlc compile ${@D}/CoremlEncoder.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlEncoder.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/encoderWrapper.mm ${@D}/CoremlEncoder.m -o ${@D}/encoderWrapper.so -framework Foundation -framework CoreML -framework Accelerate

small/decoderTest: decoderTest.cpp small/decoderWrapper.so
	cp decoderTest.cpp ${@D}
	c++ -I${@D} ${@D}/decoderTest.cpp ${@D}/decoderWrapper.so -o ${@D}/decoderTest

small/decoderWrapper.so: decoderWrapper.h decoderWrapper.mm
	cp decoderWrapper* ${@D}
	xcrun coremlc compile ${@D}/CoremlDecoder.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/decoderWrapper.mm ${@D}/CoremlDecoder.m -o ${@D}/decoderWrapper.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

small/decoder256Test: decoder256Test.cpp small/decoder256Wrapper.so
	cp decoder256Test.cpp ${@D}
	c++ -I${@D} ${@D}/decoder256Test.cpp ${@D}/decoder256Wrapper.so -o ${@D}/decoder256Test

small/decoder256Wrapper.so: decoder256Wrapper.h decoder256Wrapper.mm
	cp decoder256Wrapper* ${@D}
	xcrun coremlc compile ${@D}/CoremlDecoder256.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder256.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/decoder256Wrapper.mm ${@D}/CoremlDecoder256.m -o ${@D}/decoder256Wrapper.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

clean:
	rm -f small/decoderWrapper.so
	rm -rf small/CoremlDecoder256.mlmodelc
	rm -f small/CoremlDecoder256.h
	rm -f small/CoremlDecoder256.m
	rm -f small/decoderTest
	rm -rf small/CoremlDecoder.mlmodelc
	rm -f small/CoremlDecoder.h
	rm -f small/CoremlDecoder.m
	rm -f small/decoder256Wrapper.so
	rm -f small/decoder256Test
	rm -f small/encoderWrapper.so
