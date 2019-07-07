#!/bin/sh

mkdir base-networks
mkdir from-paper

fileid=1CMpgUL3wgas4TTUqJi8JQsHH7gW3pag9
filename=from-paper/CMU-CS-Vistas-CE.pth
if ! [ -f "$filename" ]; then 
  gdrive_download "$fileid" "$filename"
fi



