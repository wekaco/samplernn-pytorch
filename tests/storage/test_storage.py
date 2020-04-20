import pytest
from storage import storage

@pytest.mark.asyncio
async def test_upload_return_on_empty_bucket():
	with pytest.raises(ValueError, match="Bucket value must not be empty"):
		await storage.upload(None, 'file')

@pytest.mark.asyncio
async def test_upload_success():
	with pytest.raises(ValueError, match="Bucket value must not be empty"):
		await storage.upload('bucket', 'file')
		


