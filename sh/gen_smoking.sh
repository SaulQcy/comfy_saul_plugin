#!/bin/bash
echo "$1"
echo "$2"

PEOPLE_FOLDER="$1"
SIGARERTTE_FOLDER="$2"

find "$PEOPLE_FOLDER" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | while read -r img; do
    echo "$img"
    base="${img%.*}"  # 去掉扩展名
    json_file="${base}.json"
    if [ -f "$json_file" ] && grep -q '"label": "smoke",' "$json_file"; then
        # already have cigarette
        bash ./gen_smoking_single.sh $img

    else
        # need to patch cigarette
        echo 
        # SIGARETTE_PATH=$(find "$SIGARETTE_FOLDER" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | shuf -n 1)
        # echo $siga
        # bash ./gen_smoking_single.sh $img "$siga"
    fi
done
