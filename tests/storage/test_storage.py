import pytest
import sys
import asyncio

from storage.storage import Storage
from unittest.mock import MagicMock, Mock, patch
from asynctest import CoroutineMock

@pytest.mark.asyncio
async def test_upload_success():
    mock_blob = CoroutineMock(return_value="Uploaded")

    store = Storage(mock_blob)
    await store.upload('filename')
    mock_blob.assert_called()
    
    

    
    



