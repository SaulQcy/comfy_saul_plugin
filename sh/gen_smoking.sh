#!/bin/bash
echo "$1"
echo "$2"

PEOPLE_FOLDER="$1"
SIGARERTTE_FOLDER="$2"

find "$PEOPLE_FOLDER" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | while read -r img; do
    # echo "$img"
    if [[ "$img" == *smok* || "$img" == *cigarette* || "$img" == *Smok* ]]; then
        # already have cigarette
        echo $img
        bash ./gen_smoking_single.sh $img

    else
        # need to patch cigarette
        # bash ./gen_smoking_single.sh $img
        # siga="/home/saul/AIGC/cigarette/C43380CF0B394FFF322A7B9D69E9A65D.jpg"
        SIGARETTE_PATH=$(find "$SIGARETTE_FOLDER" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | shuf -n 1)
        echo $img
        echo $siga
        bash ./gen_smoking_single.sh $img "$siga"
    fi
done
