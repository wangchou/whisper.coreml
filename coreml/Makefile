tiny: tiny/encoderWrapper.so tiny/decoderTest tiny/decoder256Test

tiny/encoderWrapper.so: encoderWrapper.h encoderWrapper.mm
	cp encoderWrapper* ${@D}
	xcrun coremlc compile ${@D}/CoremlEncoder.mlmodel ${@D}
	xcrun coremlc generate ${@D}/CoremlEncoder.mlmodel ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/encoderWrapper.mm ${@D}/CoremlEncoder.m -o ${@D}/encoderWrapper.so -framework Foundation -framework CoreML -framework Accelerate

tiny/decoderTest: decoderTest.cpp tiny/decoderWrapper.so
	cp decoderTest.cpp ${@D}
	c++ -I${@D} ${@D}/decoderTest.cpp ${@D}/decoderWrapper.so -o ${@D}/decoderTest

tiny/decoderWrapper.so: decoderWrapper.h decoderWrapper.mm
	cp decoderWrapper* ${@D}
	xcrun coremlc compile ${@D}/CoremlDecoder.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/decoderWrapper.mm ${@D}/CoremlDecoder.m -o ${@D}/decoderWrapper.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

tiny/decoder256Test: decoder256Test.cpp tiny/decoder256Wrapper.so
	cp decoder256Test.cpp ${@D}
	c++ -I${@D} ${@D}/decoder256Test.cpp ${@D}/decoder256Wrapper.so -o ${@D}/decoder256Test

tiny/decoder256Wrapper.so: decoder256Wrapper.h decoder256Wrapper.mm
	cp decoder256Wrapper* ${@D}
	xcrun coremlc compile ${@D}/CoremlDecoder256.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder256.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/decoder256Wrapper.mm ${@D}/CoremlDecoder256.m -o ${@D}/decoder256Wrapper.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore

clean:
	rm -f tiny/decoderWrapper.so
	rm -f tiny/decoderTest
	rm -f tiny/decoder256Wrapper.so
	rm -f tiny/decoder256Test
	rm -f tiny/encoderWrapper.so
