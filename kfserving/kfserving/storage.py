# Copyright 2019 kubeflow.org.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import logging
import os
import tempfile

from minio import Minio
from urlparse import urlparse

_S3_PREFIX = "s3://"
_LOCAL_PREFIX = "file://"


class Storage(object): # pylint: disable=too-few-public-methods
    @staticmethod
    def download(uri, out_dir = None):
        logging.info("Copying contents of %s to local", uri)

        is_local = False
        if uri.startswith(_LOCAL_PREFIX) or os.path.exists(uri):
            is_local = True

        if out_dir is None:
            if is_local:
                # noop if out_dir is not set and the path is local
                return Storage._download_local(uri)
            out_dir = tempfile.mkdtemp()

        if uri.startswith(_S3_PREFIX):
            Storage._download_s3(uri, out_dir)
        elif is_local:
            return Storage._download_local(uri, out_dir)
        else:
            raise Exception("Cannot recognize storage type for " + uri +
                            "\n'%s', and '%s' are the current available storage type." %
                            (_S3_PREFIX, _LOCAL_PREFIX))

        logging.info("Successfully copied %s to %s", uri, out_dir)
        return out_dir

    @staticmethod
    def _download_s3(uri, temp_dir):
        client = Storage._create_minio_client()
        bucket_args = uri.replace(_S3_PREFIX, "", 1).split("/", 1)
        bucket_name = bucket_args[0]
        bucket_path = bucket_args[1] if len(bucket_args) > 1 else ""
        objects = client.list_objects(bucket_name, prefix=bucket_path, recursive=True)
        count = 0
        for obj in objects:
            # Replace any prefix from the object key with temp_dir
            subdir_object_key = obj.object_name.replace(bucket_path, "", 1).strip("/")
            # fget_object handles directory creation if does not exist
            if not obj.is_dir:
                if subdir_object_key == "":
                    subdir_object_key = obj.object_name
                client.fget_object(bucket_name, obj.object_name,
                                   os.path.join(temp_dir, subdir_object_key))
            count = count + 1
        if count == 0:
            raise RuntimeError("Failed to fetch model. The path or model %s does not exist." % (uri))

    @staticmethod
    def _download_local(uri, out_dir=None):
        local_path = uri.replace(_LOCAL_PREFIX, "", 1)
        if not os.path.exists(local_path):
            raise RuntimeError("Local path %s does not exist." % (uri))

        if out_dir is None:
            return local_path
        elif not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        if os.path.isdir(local_path):
            local_path = os.path.join(local_path, "*")

        for src in glob.glob(local_path):
            _, tail = os.path.split(src)
            dest_path = os.path.join(out_dir, tail)
            logging.info("Linking: %s to %s", src, dest_path)
            os.symlink(src, dest_path)
        return out_dir

    @staticmethod
    def _create_minio_client():
        # Remove possible http scheme for Minio
        url = urlparse(os.getenv("AWS_ENDPOINT_URL", "s3.amazonaws.com"))
        use_ssl = url.scheme == 'https' if url.scheme else bool(os.getenv("S3_USE_HTTPS", "true"))
        return Minio(url.netloc,
                     access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
                     secret_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
                     secure=use_ssl)
