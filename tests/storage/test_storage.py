import pytest
import sys
from unittest.mock import MagicMock, patch
from asynctest import CoroutineMock

# sys.modules['Blob'] = MagicMock()
# sys.modules['Blob'].__upload_from_filename = MagicMock(return_value="weeeeee")

from storage import storage

@pytest.mark.asyncio
async def test_upload_return_on_empty_bucket():
    with pytest.raises(ValueError, match="Bucket value must not be empty"):
        await storage.upload(None, 'file')

@pytest.mark.asyncio
async def test_upload_success():
    with patch('gcloud.aio.storage.blob.upload', new=CoroutineMock()) as mocked_upload:
        await storage.upload('bucket', 'file')
        mocked_upload_from_filename.assert_called_once()
    
    



