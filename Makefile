#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = simcoder-pytorch
PWD := $(shell pwd)

# docker options
DOCKER_IMAGE_NAME = simcoder-pytorch

# docker build process info
LOCAL_USER = $(USER)
LOCAL_UID = $(shell id -u)
LOCAL_GID = $(shell id -g)

#################################################################################
# CONTAINER COMMANDS                                                            #
#################################################################################
docker_image:
	docker build -t $(DOCKER_IMAGE_NAME) .

docker_run_elnuevo_interactive:
	docker run \
		--rm \
		--gpus all \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME) \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
		-v /home/$(LOCAL_USER)/development/${PROJECT_NAME}:/workspace/${PROJECT_NAME} \
		-v /home/$(LOCAL_USER)/datasets/mf/images:/input \
		-v /home/$(LOCAL_USER)/models:/models \
		-v /home/$(LOCAL_USER)/results/similarity:/output \
		-it $(DOCKER_IMAGE_NAME):latest

docker_run_tars_interactive:
	docker run \
		--rm \
		--gpus all \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME)  \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
		-v /home/$(LOCAL_USER)/development/simcoder-pytorch:/workspace/${PROJECT_NAME} \
		-v /scratch/dm236/mf/images:/input \
		-v /scratch/dm236/mf_embeddings:/output \
		-it $(DOCKER_IMAGE_NAME):latest

docker_run_dada_interactive:
	docker run \
		--rm \
		--gpus all \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME) \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
		-v /home/$(LOCAL_USER)/repos/simcoder-pytorch:/workspace/repos/$(PROJECT_NAME) \
		-v /Volumes/Data:/Volumes/Data \
		-it $(DOCKER_IMAGE_NAME):latest

run_batch:
	docker run \
		--rm \
		--gpus all \
		-u $(LOCAL_UID):$(LOCAL_GID) \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME) \
		-v /home/$(LOCAL_USER)/datasets/mf/images:/input \
		-v /home/$(LOCAL_USER)/results/similarity:/output \
		-it $(DOCKER_IMAGE_NAME):latest \
		/input /output --format=csv --chunksize=10000

# python simcoder /input /output/mf_alexnet_fc6 alexnet_fc6 128 --dirs --format=csv
# --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \

# Don't need to run anything - will run notebooks interactively.
