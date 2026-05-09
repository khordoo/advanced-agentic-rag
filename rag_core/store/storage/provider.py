from abc import ABC,abstractmethod
import os
import logging
from urllib.parse import urlparse
from urllib.request import url2pathname
from pathlib import Path
from typing import BinaryIO, Generator, Iterator
import contextlib

logger = logging.getLogger(__name__)

class SourceProvider(ABC):
    @abstractmethod
    @contextlib.contextmanager
    def open_stream(self,uri:str) -> Generator[BinaryIO, None, None]:
        """Yields a readable byte stream for the given URI."""
        ...
    @abstractmethod
    def iter_assets(self,base_uri:str)-> Iterator[str]:
        """Yields all raw file URIs found under the base URI, without filtering."""
        ...
    @abstractmethod
    def get_access_url(self, uri: str) -> str:
        """Translates a storage URI into a browser-clickable URL."""
        ...

    @property
    @abstractmethod
    def source_system(self) ->str:
        """The canonical name of the origin system (e.g., 'local_fs', 'aws_s3', 'zendesk_api')."""
        ...


class LocalFileSourceProvider(SourceProvider):
    @contextlib.contextmanager
    def open_stream(self,uri:str)-> Generator[BinaryIO, None, None]:
        with open(uri, 'rb') as f:
            yield f

    def iter_assets(self,base_uri:str)-> Iterator[str]:
        """Yields all raw file URIs found under the base URI, without filtering."""
        logger.info("Started Iteratng files in base uri: %s",base_uri)
        parsed = urlparse(base_uri)
        # 2. Extract the actual OS path
        if parsed.scheme == "file":
            # url2pathname safely converts URL paths to OS paths
            # (crucial for Windows drive letters like /C:/data -> C:\data)
            # Catch the relative path trap
            if parsed.netloc == ".":
                # Reconstruct the relative path: "." + "/data/..." -> "./data/..."
                local_path = "." + parsed.path
            else:
                # Normal absolute file paths (e.g., file:///Users/me/...)
                local_path = url2pathname(parsed.path)
        else:
            # If no scheme was provided (e.g. "./data"), trust the raw string
            local_path = base_uri
        for root, dirs, files in os.walk(local_path, topdown=True):
            # Weprune massive hidden dirs like .git to save I/O time
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                yield os.path.join(root, file)
        logger.info("Completed iteratng files in base uri: %s",base_uri)
    def get_access_url(self, uri: str) -> str:
            # Ensure it's an absolute path, then prepend the file protocol
            absolute_path = str(Path(uri).resolve())
            return f"file://{absolute_path}"
    @property
    def source_system(self):
        return "local_fs"



class S3SourceProvider(SourceProvider):
    @contextlib.contextmanager
    def open_stream(self,uri:str)-> Generator[BinaryIO, None, None]:
        NotImplemented("Future RoadMap. Please Implement")

    def iter_files(self,base_uri:str)-> Iterator[str]:
        """Yields all raw file URIs found under the base URI, without filtering."""
        NotImplemented("Future RoadMap. Please Implement")

    def get_access_url(self, uri: str) -> str:
            # uri = "s3://bucket-name/docs/report.pdf"
            # bucket, key = self._parse_s3_uri(uri)
            # # Ask AWS for a temporary link that expires in 1 hour
            # presigned_url = self.s3_client.generate_presigned_url(
            #     'get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=3600
            # )
            # return presigned_url
            NotImplemented(...)
