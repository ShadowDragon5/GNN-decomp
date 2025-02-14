#!/bin/sh

NAME=gnndecomp:24.10
podman build -t $NAME . && \
podman image ls && \
enroot import -x mount -o "$(echo $NAME | tr ':' '-')".sqsh podman://$NAME
