import hashlib
from typing import BinaryIO

def generate_file_id(source_uri:str, limit=8) -> str:
    """ Generates a sematically meaninfull unique name for the file."""
    hash=hashlib.md5(source_uri.encode('utf-8'))
    hash=hash.hexdigest()
    _, extension = source_uri.rsplit('.',1)
    _,parent,full_file_name =  source_uri.rsplit('/',2)
    file_name ,extension = full_file_name.split('.')
    file_id = f"{parent}_{file_name}_{hash[:limit]}.{extension}".lower()
    return file_id

def compute_content_hash(stream_manager: BinaryIO, algorithm='sha256'):
    with stream_manager as stream:
        hash= hashlib.file_digest(stream,algorithm)
        return hash.hexdigest()
