#!/usr/bin/env bash
if [[ -z "$1" ]];
then
  LOCAL_PYTHON_VERSION=36
elif [ "$1" == "27" ] || [ "$1" == "35" ] || [ "$1" == "36" ];
then
  LOCAL_PYTHON_VERSION=$1
else
  echo "[pyMulticopterSim] Wrong python version. Use (27/35/36)"
  return
fi
echo "python version $LOCAL_PYTHON_VERSION"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd ${DIR}/../
if [ "$LOCAL_PYTHON_VERSION" == "27" ];
then
  python2.7 -m virtualenv venv_py27
  source venv_py27/bin/activate
elif [ "$LOCAL_PYTHON_VERSION" == "35" ];
then
  python3.5 -m virtualenv venv_py35
  source venv_py35/bin/activate
elif [ "$LOCAL_PYTHON_VERSION" == "36" ];
then
  python3.6 -m virtualenv venv_py36
  source venv_py36/bin/activate
else
  return
fi

if [[ "$VIRTUAL_ENV" != "" ]]
then
  pip install -r "$DIR/requirements.txt"
else
  echo "[pyTrajectoryUtils] activate virtualenv before installing requirements"
fi

