#!/bin/sh

NAME=gnndecomp:25.01
podman build -t $NAME . && \
podman image ls && \
enroot import -x mount -o gnndecomp-25.01.sqsh podman://$NAME
