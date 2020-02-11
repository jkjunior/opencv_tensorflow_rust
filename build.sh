#!/usr/bin/env bash
mkdir /tmp/libs
cd /tmp/libs

apt download $(apt-rdepends libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libatlas-base-dev gfortran | grep -v "^ " \
| grep -v "^debconf\|libc-dev\|libz-dev\|fonts-freefont\|dbus-session-bus\|default-dbus-session-bus\|gsettings-backend")

for f in `ls -1 *.deb | sed 's/\(.*\)\..*/\1/'`
  do
  dpkg-deb -xv $f.deb .
  rm $f.deb
  done
