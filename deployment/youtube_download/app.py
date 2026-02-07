import concurrent.futures
import io
import os
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
SECONDS_TO_TIMEOUT = 30
load_dotenv()


s3 = boto3.resource("s3")

s3_client = boto3.client("s3")

import signal


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


def download_clip_s3(id, base_path):
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
    s3.meta.client.upload_file(
        output_path, AWS_S3_BUCKET, f"full_audio/data/{id}.wav"
    )
    os.remove(output_path)


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


def parallel_download(
    dataframe,
    base_path,
    files_downloaded,
    batch_size=8,
    indexes_to_use=None,
):
    if indexes_to_use is None:
        index = dataframe.index
    else:
        index = indexes_to_use
    for i in tqdm(range(0, len(index), batch_size)):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=batch_size
        ) as executor:
            futures = []
            for j in range(batch_size):
                if j + i >= len(index):
                    break
                current_id = dataframe["youtube_id"][index[i + j]]

                if current_id in files_downloaded:
                    print(f"{current_id} already downloaded skipping!")
                    dataframe["downloaded"][index[i + j]] = True
                elif not pd.isnull(current_id):
                    futures.append(
                        executor.submit(
                            download_clip_s3,
                            f"{current_id}",
                            base_path=base_path,
                        )
                    )
                else:
                    print(current_id)

            # Wait for all tasks to complete
            if len(futures) == 0:
                continue
            concurrent.futures.wait(futures, timeout=SECONDS_TO_TIMEOUT)
            for j, future in enumerate(futures):
                try:
                    future.result()
                    result = True
                except Exception as e:
                    print(e)
                    result = False
                    print(index[i + j], result, end=" ")
                dataframe["downloaded"][index[i + j]] = result
            print("...")


if __name__ == "__main__":
    files_to_process = list_bucket("msd_youtube_ids/")
    if not os.path.exists("data/"):
        os.makedirs("data/")
    for file_ in files_to_process:
        file_basename = file_.split("/")[-1].split(".")[0]
        files_locked = list_bucket("temp/")
        files_locked = [
            file.split("/")[-1].split(".")[0].replace("-", "/") + ".tsv"
            for file in files_locked
        ]
        files_processed = list_bucket("full_audio/metadata/")
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
            files_downloaded = list_bucket("full_audio/data/")
            files_downloaded = [
                file_.split("/")[-1].split(".")[0]
                for file_ in files_downloaded
            ]
            dataframe = read_dataframe_s3(file_)
            if dataframe is None:
                print(f"{file_} not found!")
                break
            dataframe["downloaded"] = False
            batch_size = 4
            parallel_download(
                dataframe,
                indexes_to_use=None,
                base_path="data/",
                files_downloaded=files_downloaded,
            )
            new_path = "full_audio/metadata/" + file_basename + ".tsv"
            write_dataframe_s3(dataframe, new_path)
        except Exception as e:
            traceback.print_exc()
        remove_lock(file_)
