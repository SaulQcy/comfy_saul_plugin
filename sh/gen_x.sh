#!/bin/bash
echo "$1"
echo "$2"

PEOPLE_FOLDER="$1"
X="$2"  # smoke or cup

find "$PEOPLE_FOLDER" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | while read -r img; do
    # echo "$img"
    # base="${img%.*}"  # 去掉扩展名
    # json_file="${base}.json"
    # if [ -f "$json_file" ] && grep -q "\"label\": \"$X\"," "$json_file"; then
    #     # already have cigarette
    #     bash ./gen_x_single.sh $img $X
    # fi
    bash ./gen_x_single.sh $img $X
done
