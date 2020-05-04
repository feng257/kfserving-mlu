import argparse
import os
import tempfile
from urllib.parse import urlparse

from minio import Minio

_S3_PREFIX = "s3://"


def download_s3(uri, access_key, secret_key, out_dir=None):
    s3_url = urlparse(uri).netloc
    S3_PREFIX = _S3_PREFIX + s3_url + "/"
    client = Minio(s3_url, access_key=access_key, secret_key=secret_key, secure=False)
    bucket_args = uri.replace(S3_PREFIX, "", 1).split("/", 1)
    bucket_name = bucket_args[0]
    bucket_path = bucket_args[1] if len(bucket_args) > 1 else ""
    objects = client.list_objects(bucket_name, prefix=bucket_path, recursive=True)
    print(s3_url, bucket_name, bucket_path, access_key, secret_key)
    count = 0
    for obj in objects:
        # Replace any prefix from the object key with temp_dir
        subdir_object_key = obj.object_name.replace(bucket_path, "", 1).strip("/")
        # fget_object handles directory creation if does not exist
        if not obj.is_dir:
            if subdir_object_key == "":
                subdir_object_key = obj.object_name
            os.path.join(out_dir, subdir_object_key)
            client.fget_object(bucket_name, obj.object_name,
                               os.path.join(out_dir, subdir_object_key))
        count = count + 1
    if count == 0:
        raise RuntimeError("Failed to fetch model. The path or model %s does not exist." % (uri))
    return out_dir


def check_savedmodel_dir(saved_model_dir):

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", help="the uri of s3 file")
    parser.add_argument("--access_key", help="s3 access key")
    parser.add_argument("--secret_key", help="s3 secret key")
    parser.add_argument("--output_dir", help="download dir")
    args = parser.parse_args()

    if not args.access_key:
        args.access_key = os.getenv("ACCESS_KEY_ID", "")
    if not args.secret_key:
        args.secret_key = os.getenv("SECRET_ACCESS_KEY", "")
    if not args.uri:
        args.uri = os.getenv("URI", "")
    if not args.output_dir:
        args.output_dir = os.getenv("OUTPUT_DIR", "")
        if not args.output_dir:
            args.output_dir=tempfile.mkdtemp()

    download_dir = download_s3(args.uri, args.access_key, args.secret_key, args.output_dir)
    '''
    python ./minio-client.py --uri=s3://10.42.127.58:9000/models/inception_v3 \
      --access_key=AKIAIOSFODNN7EXAMPLE \
      --secret_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
      --output_dir=/data/models/inception_v3
    '''
