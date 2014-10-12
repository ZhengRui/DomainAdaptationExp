#!/bin/bash

# create directories 
for Folder in `find "./images/" -type d -not -path "./images/amazon*"`
do
    NewFolder30x30=${Folder/images/thumbnails\/30x30}
    echo "mkdir: "$NewFolder30x30
    mkdir -p $NewFolder30x30
done

# convert images
for Img in `find "./images/" -name "*.jpg" -not -path "./images/amazon/*"`
do
    NewImg30x30=${Img/images/thumbnails\/30x30} 
    echo "Resizing image: "$NewImg30x30
    convert -resize 30x30\! $Img $NewImg30x30
done
