clean:
	rm -r my-hls-test
	rm my-hls-test.tar.gz

build:
	hls4ml build -p my-hls-test -c


convert:
	hls4ml convert -c keras-config.yml

synth:
	hls4ml build -p my-hls-test -s

co-sim:
	hls4ml build -p my-hls-test -r