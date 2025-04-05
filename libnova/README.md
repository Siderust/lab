#### Install libnova

```
apt-get update
apt-get install autoconf automake libtool
./autogen.sh
./configure
make
sudo make install

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

gcc -o precission.a precisson.c -lnova -lm
./precission.a
```
