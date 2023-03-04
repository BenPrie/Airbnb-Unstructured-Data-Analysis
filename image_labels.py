import boto3
import urllib.request
import ssl
import re

# Proof of concept
# These methods can be used to generate image labels for any of the images posted on an Airbnb listing

def get_listing_pictures(listing_url):
    pictures = []

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(listing_url, context=ctx) as f:
        html = f.read().decode('utf-8')
    pattern = r'https:\/\/a0\.muscache\.com\/pictures\/\d+\/\w+\.jpg'
    matches = re.findall(pattern, html)
    pictures += list(set(matches))
    pattern = r'https:\/\/a0\.muscache\.com\/pictures\/[-\w]+\.jpg'
    matches = re.findall(pattern, html)
    pictures += list(set(matches))
    return pictures

def get_labels(url):
    ACCESS_KEY = '' # disabled key
    SECRET_KEY = '' # disabled key

    client = boto3.client(
        'rekognition',
        aws_access_key_id = ACCESS_KEY,
        region_name = 'us-west-2',
        aws_secret_access_key = SECRET_KEY)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(url, context=ctx) as f:
        image = f.read()
        return client.detect_labels(Image={'Bytes':image})

if __name__ == '__main__':

    for picture in get_listing_pictures('https://www.airbnb.com/rooms/10030323'): # to be replaced with df[
        # listing_url] where df is the csv.gv file for any city in the dataset
        labels = get_labels(picture)
        print([l['Name'] for l in labels['Labels']])
