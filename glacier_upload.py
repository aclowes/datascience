import sys
import os.path

from boto.glacier.layer1 import Layer1
from boto.glacier.vault import Vault
from boto.glacier.concurrent import ConcurrentUploader

access_key_id = None
secret_key = None
target_vault_name = 'backups'
region_name = 'us-west-2'
fname = sys.argv[1]

if (os.path.isfile(fname) == False):
    print("Can't find the file to upload!");
    sys.exit(-1);


glacier_layer1 = Layer1(aws_access_key_id=access_key_id, aws_secret_access_key=secret_key,
                        region_name=region_name)

uploader = ConcurrentUploader(glacier_layer1, target_vault_name, 32 * 1024 * 1024)

print("operation starting...");

archive_id = uploader.upload(fname, fname)

print("Success! archive id: '%s'" % (archive_id))