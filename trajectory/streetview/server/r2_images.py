"""Upload immutable candidate JPEGs, verifying every public URL before Batch POST."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import requests
from PIL import Image

from common import SERVER_ROOT, confined, load
from stage_store import durable_save
from batch_tunnels import image_parts, TunnelUnavailable


class R2ConfigurationError(RuntimeError):
    """Credentials, permissions, or immutable-object conflicts need attention."""


def load_settings(run, transport):
    settings = load(confined(SERVER_ROOT / transport.get('config_file', 'config/r2.json')))
    endpoint = settings.get('endpoint', '').rstrip('/')
    if not re.fullmatch(r'https://[0-9a-f]{32}(?:\.(?:eu|fedramp))?\.r2\.cloudflarestorage\.com', endpoint):
        raise R2ConfigurationError('R2 requires a complete Cloudflare S3 endpoint')
    base = settings.get('public_base_url', '').rstrip('/')
    parsed = urlsplit(base)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.path or parsed.query or parsed.fragment or parsed.port):
        raise R2ConfigurationError('R2 public base must be an HTTPS origin')
    if transport.get('base_url', base).rstrip('/') != base:
        raise R2ConfigurationError('R2 public base differs from dataset transport')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,61}[a-z0-9]', settings.get('bucket', '')):
        raise R2ConfigurationError('Invalid R2 bucket name')
    prefix = settings.get('key_prefix', run.name + '/candidates').strip('/')
    if not prefix or any(not re.fullmatch(r'[A-Za-z0-9_-]+', p) for p in prefix.split('/')):
        raise R2ConfigurationError('Invalid R2 object prefix')
    workers = settings.get('upload_workers', 16)
    if type(workers) is not int or not 1 <= workers <= 32:
        raise R2ConfigurationError('R2 upload workers must be 1..32')
    return {**settings, 'endpoint': endpoint, 'public_base_url': base,
            'key_prefix': prefix, 'upload_workers': workers}


def create_client(settings):
    # Explicit credentials avoid SDK searches in home directories or metadata services.
    from boto3.session import Session
    from botocore.config import Config
    p = confined(SERVER_ROOT / settings.get('credentials_file', 'config/r2_credentials.json'))
    if os.name != 'nt' and p.stat().st_mode & 0o077:
        raise R2ConfigurationError('R2 credential file must have mode 0600')
    secret = load(p)
    if not all(isinstance(secret.get(k), str) and secret[k].strip()
               for k in ('access_key_id', 'secret_access_key')):
        raise R2ConfigurationError('R2 credentials are incomplete')
    return Session(aws_access_key_id=secret['access_key_id'],
                   aws_secret_access_key=secret['secret_access_key'], region_name='auto').client(
        's3', endpoint_url=settings['endpoint'],
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'},
                      connect_timeout=10, read_timeout=45, proxies={},
                      max_pool_connections=settings['upload_workers'],
                      retries={'mode': 'standard', 'total_max_attempts': 3},
                      request_checksum_calculation='when_required',
                      response_checksum_validation='when_required'))


def error_code(error):
    return str(getattr(error, 'response', {}).get('Error', {}).get('Code', ''))


def upload_verified(client, settings, city, filename, data):
    """HEAD/reuse or create-only PUT, then fetch the exact bytes from public HTTPS."""
    from botocore.exceptions import BotoCoreError, ClientError
    sha = hashlib.sha256(data).hexdigest()
    key = settings['key_prefix'] + '/' + city + '/' + filename
    bucket = settings['bucket']

    def inspect_existing():
        try:
            info = client.head_object(Bucket=bucket, Key=key)
        except ClientError as error:
            if error_code(error) in ('404', 'NoSuchKey', 'NotFound'):
                return False
            raise
        if info.get('Metadata', {}).get('sha256') != sha or info.get('ContentLength') != len(data):
            raise R2ConfigurationError('R2 immutable object conflicts with local candidate bytes')
        return True

    try:
        existed = inspect_existing()
        if not existed:
            try:
                client.put_object(Bucket=bucket, Key=key, Body=data, ContentType='image/jpeg',
                                  CacheControl='public, max-age=31536000, immutable',
                                  Metadata={'sha256': sha}, IfNoneMatch='*')
            except ClientError as error:
                if error_code(error) not in ('412', 'PreconditionFailed') or not inspect_existing():
                    raise
    except ClientError as error:
        code = error_code(error)
        if code in ('AccessDenied', 'InvalidAccessKeyId', 'SignatureDoesNotMatch',
                    'InvalidToken', 'NoSuchBucket', '403', '401'):
            raise R2ConfigurationError('R2 authentication/bucket permission rejected: ' + code) from None
        raise TunnelUnavailable('R2 object API unavailable; no provider POST attempted') from None
    except BotoCoreError:
        raise TunnelUnavailable('R2 object network unavailable; no provider POST attempted') from None
    url = settings['public_base_url'] + '/' + key
    with requests.Session() as session:
        session.trust_env = False
        for attempt in range(3):
            try:
                response = session.get(url, timeout=(5, 20), allow_redirects=False)
                if response.status_code == 200 and response.content == data:
                    return existed
            except requests.RequestException:
                pass
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise TunnelUnavailable('R2 public JPEG byte verification failed; no provider POST attempted')


def prepare_r2(run, city, items, transport):
    settings = load_settings(run, transport)
    if not re.fullmatch(r'[a-z0-9_]+', city):
        raise ValueError('Invalid R2 city')
    outgoing = deepcopy(items)
    files = set()
    images = list(image_parts(outgoing))
    for image in images:
        parsed = urlsplit(image['url'])
        if (parsed.scheme not in ('http', 'https') or parsed.query or parsed.fragment
                or not re.fullmatch('/' + re.escape(city) + r'/[0-9a-f]{64}\.jpg', parsed.path)):
            raise ValueError('R2 candidate path does not match its city')
        filename = parsed.path.rsplit('/', 1)[-1]
        files.add(filename)
    # Validate everything before rewriting: deepcopy preserves shared references.
    for image in images:
        filename = urlsplit(image['url']).path.rsplit('/', 1)[-1]
        image['url'] = (settings['public_base_url'] + '/' + settings['key_prefix']
                        + '/' + city + '/' + filename)
    # Only a city-level upload summary is persisted; no per-point upload records.
    state_path = confined(run / 'state' / ('r2_upload_' + city + '.json'))
    summary = {'city_id': city, 'transport': 'r2', 'bucket': settings['bucket'],
               'base_url': settings['public_base_url'], 'key_prefix': settings['key_prefix'],
               'request_count': len(items), 'image_count': len(files), 'verified_images': 0,
               'reused_images': 0, 'status': 'uploading', 'started_at': time.time()}
    durable_save(state_path, summary)
    client = None
    directory = confined(run / 'work' / city / 'public_images')

    def upload(filename):
        data = confined(directory / filename).read_bytes()
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != 'JPEG':
                    raise ValueError('Not a JPEG')
                image.verify()
        except Exception:
            raise R2ConfigurationError('Local R2 candidate is not a valid JPEG') from None
        return upload_verified(client, settings, city, filename, data)

    try:
        client = create_client(settings)
        with ThreadPoolExecutor(max_workers=settings['upload_workers'], thread_name_prefix='r2') as pool:
            filenames = iter(sorted(files))
            pending = set()
            for _ in range(settings['upload_workers']):
                filename = next(filenames, None)
                if filename is not None:
                    pending.add(pool.submit(upload, filename))
            last_saved = time.monotonic()
            try:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    # Process every finished result before enqueueing more uploads.
                    for future in done:
                        summary['reused_images'] += int(future.result())
                        summary['verified_images'] += 1
                    for _ in done:
                        filename = next(filenames, None)
                        if filename is not None:
                            pending.add(pool.submit(upload, filename))
                    if time.monotonic() - last_saved >= 20:
                        summary['updated_at'] = time.time()
                        durable_save(state_path, summary)
                        last_saved = time.monotonic()
            finally:
                for future in pending:
                    future.cancel()
        summary.update(status='verified', verified_at=time.time())
        durable_save(state_path, summary)
    except Exception as error:
        summary.update(status='blocked', error_type=type(error).__name__, updated_at=time.time())
        durable_save(state_path, summary)
        raise
    finally:
        if client is not None:
            client.close()
    return outgoing, {k: summary[k] for k in ('bucket', 'base_url', 'key_prefix',
                                             'request_count', 'image_count', 'verified_images')}
