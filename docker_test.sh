#!/bin/sh

docker run -i debian <<EOF
apt update >/dev/null 
apt install -y pipx git  >/dev/null 
yes | adduser --quiet test
su test
pipx install git+https://github.com/pixelsoup42/cardano_account_pandas_dumper
pipx ensurepath
bash -l -c 'cardano_account_pandas_dumper -h'
EOF