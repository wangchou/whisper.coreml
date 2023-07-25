#usage: model=[tiny|base|small|medium|large] make [all|clean]
MODEL = $$model

all: $(MODEL)/coreml.so

$(MODEL)/coreml.so: coreml.h coreml.mm
	cp -f *.h ${@D}
	cp -f *.mm ${@D}
	for mlmodel in $(shell find $(MODEL) -name '*.mlpackage') ; do \
		xcrun coremlc generate $$mlmodel ${@D} || exit 1; \
		xcrun coremlc compile  $$mlmodel ${@D} || exit 1; \
    done

	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/coreml.mm ${@D}/coremlUtility.mm ${@D}/CoremlEncoder0.m ${@D}/CoremlCrossKV.m ${@D}/CoremlDecoder.m ${@D}/CoremlDecoder256.m -o ${@D}/coreml.so -framework Foundation -framework CoreML -framework Accelerate

clean:
	rm -f $(MODEL)/*.so
