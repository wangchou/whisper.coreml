#usage: model=[tiny|base|small|medium|large] make [all|clean]
MODEL = $$model

all: $(MODEL)/coremlTest

$(MODEL)/coremlTest: coremlTest.cpp $(MODEL)/coreml.so
	cp coremlTest.cpp ${@D}
	c++ -I${@D} ${@D}/coremlTest.cpp ${@D}/coreml.so -o ${@D}/coremlTest

$(MODEL)/coreml.so: coreml.h coreml.mm
	cp -f *.h ${@D}
	cp -f *.mm ${@D}
	for mlmodel in $(shell find $(MODEL) -name '*.mlpackage') ; do \
		xcrun coremlc generate $$mlmodel ${@D} || exit 1; \
		xcrun coremlc compile  $$mlmodel ${@D} || exit 1; \
		rm -rf $$mlmodel;\
    done

	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/coreml.mm ${@D}/coremlUtility.mm ${@D}/Encoder0.m ${@D}/CrossKV.m ${@D}/Decoder.m ${@D}/Decoder256.m -o ${@D}/coreml.so -framework Foundation -framework CoreML -framework Accelerate

clean:
	rm -f $(MODEL)/*.so
	rm -f $(MODEL)/coremlTest
