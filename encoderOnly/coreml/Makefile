#usage: model=[tiny|base|small|medium|large] make [all|clean]
MODEL = $$model

all: $(MODEL)/encoderTest $(MODEL)/decoderTest $(MODEL)/decoder256Test

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

clean:
	rm -rf $(MODEL)/*.mlmodelc
	rm -f $(MODEL)/*.h
	rm -f $(MODEL)/*.m
	rm -f $(MODEL)/*.mm
	rm -f $(MODEL)/*.cpp
	rm -f $(MODEL)/*Test
	rm -f $(MODEL)/*.so
