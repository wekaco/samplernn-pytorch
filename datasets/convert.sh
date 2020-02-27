#!/bin/sh

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <filetpath> <chunk size in seconds> <dataset path> <sample rate>"
    exit
fi

url=$1
chunk_size=$2
dataset_path=$3
rate=$4

converted=".temp.wav"
rm -f $converted
ffmpeg -i $url -ac 1 -ar $4 $converted
#rm -f $url

mkdir $dataset_path
length=$(ffprobe -i $converted -show_entries format=duration -v quiet -of csv="p=0")
end=$(echo "$length / $chunk_size - 1" | bc)
echo "splitting..."
for i in $(seq 0 $end); do
    ffmpeg -hide_banner -loglevel error -ss $(($i * $chunk_size)) -t $chunk_size -i $converted "$dataset_path/$i.wav"
done
echo "done"
rm -f $converted
