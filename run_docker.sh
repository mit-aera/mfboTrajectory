#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

docker run \
    -it \
    --publish-all \
    --rm \
    --gpus all\
    --volume "${DIR}/../mfboTrajectory:/root/mfboTrajectory" \
    --name mfbo_traj \
    --privileged \
    --net "host" \
    mfbo_traj
