import concurrent.futures
import io
import os
import signal
import traceback
import warnings

import boto3
import pandas as pd
import yt_dlp
from dotenv import load_dotenv
from tqdm import tqdm

# Suppress pandas warnings
warnings.simplefilter(
    action="ignore", category=pd.errors.SettingWithCopyWarning
)

AWS_S3_BUCKET = "idlmusicgeneration"

load_dotenv()

s3 = boto3.resource("s3")
s3_client = boto3.client("s3")


def signal_handler(signal, frame):
    print("You pressed Ctrl+C!")
    remove_lock(file_)
    exit(0)


signal.signal(signal.SIGINT, signal_handler)


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


def list_bucket(prefix):
    s3 = boto3.resource("s3")

    bucket = s3.Bucket(AWS_S3_BUCKET)

    # List objects in the bucket with the specified prefix
    results = [obj.key for obj in bucket.objects.filter(Prefix=prefix)]
    return results[1:]


def get_youtube_id(query, max_results=1):
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

    if search_results is not None and "entries" in search_results:
        try:
            video_id = search_results["entries"][0]["id"]
        except IndexError:
            print(query, "results:", search_results["entries"])
            return None
    else:
        return None
    return video_id


def create_lock_file(file):
    path = f"{file.replace('/', '-')}.lock"
    with open(path, "w") as f:
        f.write("locked")
    s3.meta.client.upload_file(path, AWS_S3_BUCKET, f"temp/{path}")


def remove_lock(file):
    path = f"{file.replace('/', '-')}.lock"
    try:
        s3.Object(AWS_S3_BUCKET, f"temp/{path}").delete()
        print("Lock removed!")
    except Exception:
        print(f"Lock for file {file} didn't exist")


def parallel_get_video_ids(dataframe, batch_size=8, indexes_to_update=None):
    if indexes_to_update is None:
        index = dataframe.index
    else:
        index = indexes_to_update
    for i in tqdm(range(0, len(index), batch_size)):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=batch_size
        ) as executor:
            # Submit download_video function for each URL in the list

            futures = [
                executor.submit(
                    get_youtube_id,
                    f"\"{dataframe['artist_name'][index[i+j]]} - {dataframe['title'][index[i+j]]}\"",
                )
                for j in range(batch_size)
                if j + i < len(index)
            ]

            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
            for j, future in enumerate(futures):
                id = future.result()
                print(index[i + j], id)
                dataframe["youtube_id"][index[i + j]] = id


if __name__ == "__main__":
    files_to_process = list_bucket("msd_processed/")
    if not os.path.exists("data/"):
        os.makedirs("data/")
    for file_ in files_to_process:
        file_basename = file_.split("/")[-1].split(".")[0]

        files_locked = list_bucket("temp/")
        files_locked = [
            file.split("/")[-1].split(".")[0].replace("-", "/") + ".tsv"
            for file in files_locked
        ]
        files_processed = list_bucket("msd_youtube_ids/")
        files_processed = [
            file_.split("/")[-1].split(".")[0] for file_ in files_processed
        ]

        if file_ in files_locked or file_basename in files_processed:
            print(f"Skipping {file_}!!!!")
            continue
        else:
            create_lock_file(file_)
            print(f"Starting process for file {file_}")

        try:
            print(f"Processing file {file_basename}")
            dataframe = read_dataframe_s3(file_)
            if dataframe is None:
                print(f"{file_} not found")
                continue

            dataframe["youtube_id"] = ""
            batch_size = 4
            parallel_get_video_ids(
                dataframe,
                indexes_to_update=None,
                batch_size=batch_size,
            )

            new_path = "msd_youtube_ids/" + file_basename + ".tsv"
            write_dataframe_s3(dataframe, new_path)
        except Exception as e:
            traceback.print_exc()
        remove_lock(file_)
