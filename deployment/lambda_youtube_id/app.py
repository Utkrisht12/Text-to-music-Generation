import io
import json
from multiprocessing import Pipe, Process

import boto3
import pandas as pd
import yt_dlp

AWS_S3_BUCKET = "idlmusicgeneration"

s3_client = boto3.client("s3")


def get_youtube_id(query, conn, max_results=1):
    # Set up yt-dlp options
    ydl_opts = {
        "extract_flat": "in_playlist",
        "max_downloads": max_results,
        "default_search": "auto",
        "quiet": True,
        "simulate": True,
    }

    # Search for videos matching the query
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        search_results = ydl.extract_info(query, download=False)

    if "entries" in search_results:  # type:ignore
        try:
            video_id = search_results["entries"][0]["id"]  # type:ignore
        except IndexError:
            print(query, "results:", search_results["entries"])  # type:ignore
            conn.send([None])
            conn.close()
            return
    else:
        conn.send([None])
        conn.close()
        return
    conn.send([video_id])
    conn.close()


def write_dataframe_s3(df, path):
    with io.StringIO() as csv_buffer:
        df.to_csv(csv_buffer, sep="\t")

        response = s3_client.put_object(
            Bucket=AWS_S3_BUCKET, Key=path, Body=csv_buffer.getvalue()
        )

        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return status


def read_dataframe_s3(path):
    response = s3_client.get_object(Bucket=AWS_S3_BUCKET, Key=path)

    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        df = pd.read_csv(response.get("Body"), sep="\t", index_col=0)
        return df
    return None


def download_clip(id, base_path):
    output_path = f"{base_path}/{id}.wav"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "audio-format": "wav",
        "quiet": True,
    }
    url = f"https://www.youtube.com/watch?v={id}"
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def parallel_get_video_ids(dataframe, batch_size=8, indexes_to_update=None):
    if indexes_to_update is None:
        index = dataframe.index
    else:
        index = indexes_to_update

    for i in range(0, len(index), batch_size):

        # Submit download_video function for each URL in the list
        if i % 7 == 0:
            print(i)
        parent_connections = []
        processes = []
        for j in range(batch_size):
            if j + i >= len(index):
                break
            parent_conn, child_conn = Pipe()
            parent_connections.append(parent_conn)

            processes.append(
                Process(
                    target=get_youtube_id,
                    args=(
                        f"\"{dataframe['artist_name'][index[i+j]]} - {dataframe['title'][index[i+j]]}\"",
                        child_conn,
                    ),
                )
            )
        # start all processes
        for process in processes:
            process.start()

        # make sure that all processes have finished
        for process in processes:
            process.join()

        # Wait for all tasks to complete
        for j, parent_connection in enumerate(parent_connections):
            id = parent_connection.recv()[0]
            # print(index[i + j], id)
            dataframe["youtube_id"][index[i + j]] = id


# {"file_name": "msd_processed/CA.tsv","batch_size": 5,"range_upd": [0,200],"debugging": true}
def lambda_handler(event, context):
    path = event["file_name"]
    batch_size = event["batch_size"]
    range_upd = event["range_upd"]
    dataframe = read_dataframe_s3(path)

    if event.get("debugging"):
        print(dataframe)

    if dataframe is None:
        return {
            "statusCode": 404,
            "body": json.dumps("ERROR, dataset not found"),
        }
    dataframe["youtube_id"] = None
    if len(range_upd) != 0:
        dataframe = dataframe.iloc[range_upd[0] : range_upd[1], :]

    parallel_get_video_ids(dataframe, batch_size=batch_size)
    new_path = (
        "msd_youtube_ids/"
        + path.split("/")[-1].split(".")[0]
        + "_"
        + "-".join([str(i) for i in range_upd])
        + ".tsv"
    )
    write_dataframe_s3(dataframe, new_path)

    return {
        "statusCode": 200,
        "body": f"Success! data saved in {new_path}",
    }
