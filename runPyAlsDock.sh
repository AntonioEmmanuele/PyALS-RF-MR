#!/bin/bash

usage() {
  echo "Usage: $0 -c catalog -s path_to_shared_folder -d dbg";
  exit 1;
}

while getopts "c:s:d:" o; do
    case "${o}" in
        c)
            catalog=${OPTARG}
            ;;
        s)
            shared=${OPTARG}
            ;;
        
        d)
            dbg=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${catalog}" ] || [ -z "${shared}" ] || [ -z "${dbg}" ]; then
    usage
fi

catalog=`realpath ${catalog}`
shared=`realpath ${shared}`
dbg=`realpath ${dbg}`
[ ! -d $shared ] && mkdir -p $shared
xhost local:docker
docker run --rm -e DISPLAY=unix$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v ${catalog}:/home/user/lut_catalog.db -v ${shared}:/home/user/shared -v ${dbg}:/home/user/pyALS-RF-dbg/ -w /home/user --privileged -it salvatorebarone/pyals-docker-image /bin/zsh
