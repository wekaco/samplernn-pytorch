build:
	docker build -t wekaco/${shell basename "${PWD}" }:${shell git rev-parse --abbrev-ref HEAD | tr '/' '_'} -f ./Dockerfile .

build9:
	docker build -t wekaco/${shell basename "${PWD}" }:${shell git rev-parse --abbrev-ref HEAD | tr '/' '_'}-cuda-9.1 -f ./Dockerfile.cuda-9.1 .

dev:
	docker run -v $(shell pwd):/app --rm -ti wekaco/${shell basename "${PWD}" }:${shell git rev-parse --abbrev-ref HEAD | tr '/' '_'}

dev9:
	docker run -v $(shell pwd):/app --rm -ti wekaco/${shell basename "${PWD}" }:${shell git rev-parse --abbrev-ref HEAD | tr '/' '_'}-cuda-9.1
