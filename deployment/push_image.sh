#!/bin/bash
#######################################
# Builds and pushes a given image to the corresponding ECR Registry
# Arguments:
#   1. image_to_push: a string that determines the image(s) to be pushed
#                     accepted values: [email/extraction/report/top_artists]
# Usage:
#    source push_image.sh report
#    source push_image.sh report top_artists
#    source push_image.sh email extraction report top_artists
#######################################


#import .env variables
export $(cat .env | xargs) &> /dev/null

accepted_values="id download trainer_audiolm"
input=$1

if [[ -z $input  ]];then
    echo "Error: please specify an image to push"
    echo "Values accepted: [$accepted_values]"
    exit 1
fi

declare -A local_image_name
local_image_name["id"]="youtube_id"
local_image_name["download"]="youtube_download"
local_image_name["trainer_audiolm"]="trainer_audiolm"

## ECR_* variables are specified in the .env file
declare -A remote_image_name
remote_image_name["id"]=$ECR_YOUTUBE_ID
remote_image_name["download"]=$ECR_YOUTUBE_DOWNLOAD
remote_image_name["trainer_audiolm"]=$ECR_TRAINER_AUDIOLM

echo $ECR_YOUTUBE_ID

(
set -e
for image_to_push in $input; do

    if ! [[ $accepted_values == *"$image_to_push"* ]]; then
        echo "Error: please specify a valid image to push"
        echo "Values accepted: [$accepted_values]"
        exit 1
    fi

    aws ecr get-login-password --region $AWS_DEFAULT_REGION | \
    docker login --username AWS --password-stdin $AWS_USER_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com
    echo "Connected to ECR"

    docker-compose build ${local_image_name[$image_to_push]}
    echo "Docker image built"

    docker tag ${local_image_name[$image_to_push]}:latest ${remote_image_name[$image_to_push]}
    echo "Docker image tagged successfully"

    docker push ${remote_image_name[$image_to_push]}
    echo ""
    echo "The ${image_to_push} docker image was succesfully pushed"
done
)
