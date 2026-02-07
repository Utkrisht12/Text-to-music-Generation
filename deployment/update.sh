files=$(aws s3 ls  s3://idlmusicgeneration/msd_processed/ | cut -d " " -f8)
max_files=1700
partition_size=80
partitions=$(echo $((max_files / partition_size)))
for file in $files; do
    file=$(echo $file | sed -e "s/'\n//g")
    for (( i=$partition_size ; i<=$max_files ; i+=$partition_size )); do
        sed -i "s/msd_processed\/.*tsv/msd_processed\/${file}/" tests/event.json
        sed -i "s/  [0-9]*,/$((i-partition_size)),/" tests/event.json
        sed -i "s/  [0-9]*$/$i/" tests/event.json
        aws lambda invoke --function-name youtube_id --payload "$(cat tests/event.json | base64)"  response.json &
    done
    break;
done