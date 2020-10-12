build:
	docker build -t wekaco/${shell basename "${PWD}" }:${shell git rev-parse --abbrev-ref HEAD | tr '/' '_'} -f ./Dockerfile .

dev:
	docker run -v $(shell pwd):/app --rm -ti wekaco/${shell basename "${PWD}" }:${shell git rev-parse --abbrev-ref HEAD | tr '/' '_'}
