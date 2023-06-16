tiny/objcWrapper.so: objcWrapper.h objcWrapper.mm
	cp obj* ${@D}
	xcrun coremlc compile ${@D}/CoremlEncoder.mlmodel ${@D}
	xcrun coremlc generate ${@D}/CoremlEncoder.mlmodel ${@D}
	clang -shared -undefined dynamic_lookup -fPIC -fobjc-arc ${@D}/objcWrapper.mm ${@D}/CoremlEncoder.m -o ${@D}/objcWrapper.so -framework Foundation -framework CoreML -framework Accelerate
clean:
	rm tiny/objcWrapper.so
