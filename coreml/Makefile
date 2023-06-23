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

small/decoderTest: decoderTest.cpp small/decoderWrapper.so
	cp decoderTest.cpp ${@D}
	c++ -I${@D} ${@D}/decoderTest.cpp ${@D}/decoderWrapper.so -o ${@D}/decoderTest

small/decoderWrapper.so: decoderWrapper.h decoderWrapper.mm
	cp decoderWrapper* ${@D}
	xcrun coremlc compile ${@D}/CoremlDecoder.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder.mlpackage ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/decoderWrapper.mm ${@D}/CoremlDecoder.m -o ${@D}/decoderWrapper.so -framework Foundation -framework CoreML -framework Accelerate -framework QuartzCore
clean:
	rm -f tiny/decoderWrapper.so
	rm -f tiny/decoderTest
	rm -f tiny/encoderWrapper.so
	rm -f small/decoderWrapper.so
	rm -f small/decoderTest
	rm -f small/encoderWrapper.so
