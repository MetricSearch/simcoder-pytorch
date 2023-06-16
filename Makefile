#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = sisap2023-pytorch
PWD := $(shell pwd)

# docker options
DOCKER_IMAGE_NAME = sisap2023

# docker build process info
LOCAL_USER = $(USER)
LOCAL_UID = $(shell id -u)
LOCAL_GID = $(shell id -g)

#################################################################################
# LOCAL ENV COMMANDS                                                            #
#################################################################################
environment:
	python -m venv venv
	. venv/bin/activate &&  pip install -r requirements.txt && pip install -e .
	echo "Remember to activate the environment before use: . venv/bin/activate"


#################################################################################
# CONTAINER COMMANDS                                                            #
#################################################################################
docker_image:
	docker build -t $(DOCKER_IMAGE_NAME) .

# run the container and attach a shell so we can run the code interactively
# useful for runnig the notebooks
interactive:
	docker run \
		--rm \
		--gpus all \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME) \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
		-v /home/$(LOCAL_USER)/repos/sisap2023:/workspace/repos/$(PROJECT_NAME) \
		-v /Volumes/Data:/Volumes/Data \
		-it $(DOCKER_IMAGE_NAME):latest

# run the default encoder to generate the dino2 embedding of mirflickr
encode:
	docker run \
		--rm \
		--gpus all \
		-u $(LOCAL_UID):$(LOCAL_GID) \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME) \
		-v /home/$(LOCAL_USER)/repos/sisap2023:/workspace/repos/$(PROJECT_NAME) \
		-v /Volumes/Data:/Volumes/Data \
		$(DOCKER_IMAGE_NAME):latest \
		python sisap2023 encode -d --format=mat /Volumes/Data/mf/images /Volumes/Data/mf_dino2 dino2 128

# generate the experimental results reported in the paper
experiment:
	docker run \
		--rm \
		--gpus all \
		-u $(LOCAL_UID):$(LOCAL_GID) \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME) \
		-v /home/$(LOCAL_USER)/repos/sisap2023:/workspace/repos/$(PROJECT_NAME) \
		-v /Volumes/Data:/Volumes/Data \
		-it $(DOCKER_IMAGE_NAME):latest \
		python sisap2023 experiment --use_preselected_queries /Volumes/Data/mf_dino2 /Volumes/Data/mf_resnet19_softmax /Volumes/Data/reported_results 100 100 0 0.9
