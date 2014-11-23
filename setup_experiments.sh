#!/bin/bash

curl https://protobuf.googlecode.com/svn/rc/protobuf-2.6.0.tar.gz
gzip -d file.gz protobuf-2.6.0.tar.gz
rm protobuf-2.6.0.tar.gz

cd ~
git clone https://github.com/JasperSnoek/spearmint.git
cd spearmint/
rm -rf spearmint-lite/
rm FAQ.md 
rm README.md 
mv spearmint todelete
cd todelete/
mv spearmint/ ..
cd ..
rm -rf todelete/

cd spearmint
python2 setup.py install --user
bin/make_protobufs
