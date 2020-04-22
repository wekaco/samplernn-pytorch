import asyncio
import os
import logging

class Storage:

    def __init__(self, blob):
        self.blob = blob

    async def upload(self, file_path):
        try:
            logging.info(f'uploading {file_path}')
            print(self.blob)
            await self.blob.upload(file_path)
        except (ConnectionAbortedError,ConnectionResetError) as conn_err:
            print(f'Connection Error: {conn_err}')
            pass
        except Exception as e:
            print(f'Exception: {e}')
            exit(1)
        return file_path


