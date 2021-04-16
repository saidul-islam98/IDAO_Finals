all: build run
	
build:
	docker build -t sidao_docker . 

run:
	docker run -it sidao_docker

.PHONY: all build run

clean: all build run