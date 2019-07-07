#!/bin/sh


# function copied from https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805
gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
    "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}


mkdir -p base-networks
mkdir -p from-paper

fileid=1CMpgUL3wgas4TTUqJi8JQsHH7gW3pag9
filename=from-paper/CMU-CS-Vistas-CE.pth
if ! [ -f "$filename" ]; then 
  gdrive_download "$fileid" "$filename"
fi

fileid=1QJy3mSXIWEvrj-GtAjcvdoIP2t1Veyeo
filename=from-paper/RC-CS-Vistas-HingeF.pth
if ! [ -f "$filename" ]; then 
  gdrive_download "$fileid" "$filename"
fi



