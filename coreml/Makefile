objcWrapper.o: objcWrapper.h objcWrapper.mm CoremlEncoder.h CoremlEncoder.m
	clang -shared -undefined dynamic_lookup -fobjc-arc objcWrapper.mm CoremlEncoder.m -o objcWrapper.o -framework Foundation -framework CoreML -framework Accelerate
clean:
	rm objcWrapper.o
