#usage: model=[tiny|base|small|medium|large] make [all|clean]
#MODEL = $$model

all: small/coreml.so

small/coreml.so: coreml.h coreml.mm
	cp *.h ${@D}
	cp *.mm ${@D}
	cp *.cpp ${@D}
	xcrun coremlc generate  ${@D}/CoremlEncoder0.mlpackage ${@D}
	for mlmodel in $(shell find small -name 'CoremlEncoder*.mlpackage') ; do \
		xcrun coremlc compile $$mlmodel ${@D} || exit 1; \
    done
	xcrun coremlc compile  ${@D}/CoremlCrossKV.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlCrossKV.mlpackage ${@D}
	xcrun coremlc compile  ${@D}/CoremlDecoder.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder.mlpackage ${@D}
	xcrun coremlc compile  ${@D}/CoremlDecoder256.mlpackage ${@D}
	xcrun coremlc generate ${@D}/CoremlDecoder256.mlpackage ${@D}

	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/coreml.mm ${@D}/coremlUtility.mm ${@D}/CoremlEncoder0.m ${@D}/CoremlCrossKV.m ${@D}/CoremlDecoder.m ${@D}/CoremlDecoder256.m -o ${@D}/coreml.so -framework Foundation -framework CoreML -framework Accelerate

small/decoder256Test: decoder256Test.cpp small/coreml.so
	cp decoder256Test.cpp ${@D}
	c++ -I${@D} ${@D}/decoder256Test.cpp ${@D}/coreml.so -o ${@D}/decoder256Test
clean:
	rm -rf small/*.mlmodelc
	rm -f small/*.h
	rm -f small/*.m
	rm -f small/*.mm
	rm -f small/*.cpp
	rm -f small/*Test
	rm -f small/*.so
