import asyncio
import os
from gcloud.aio.storage.blob import Blob
# from gcloud.aio.storage.bucket import Bucket

async def upload(bucket, file_path):
    if bucket is None:
        raise ValueError("Bucket value must not be empty")

    # remove prefix /app
    name = file_path.replace(os.path.abspath(os.curdir) + '/', '')
    blob = Blob(name, bucket)
    print(blob)
    logging.info('uploading {}'.format(name))
    try:
        await blob.upload_from_filename(file_path)
    except (ConnectionAbortedError,ConnectionResetError) as conn_err:
        print(f'Connection Error: {conn_err}')
        pass
    except Exception as e:
        print(f'Exception: {e}')
        exit(1)
    return bucket

