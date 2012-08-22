INCLUDE_PYTHON=`pkg-config --cflags python`
CFLAGS=${INCLUDE_PYTHON} -fPIC -g

compiled_sum.so: compiled_sum.o
	gcc -shared -o $@ $^

clean:
	rm compiled_sum.o compiled_sum.so