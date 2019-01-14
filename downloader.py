import magic
import os
import re
import requests
import time

URL = 'http://kern.ccarh.org/'
FORMAT = 'midi' # choose midi or kern
EXTENSIONS = {'midi': 'mid', 'kern': 'krn'}
ENCODING = 'unicode_escape'

def download():
    '''
    Download all kern files from [URL] and class them by artist
    '''
    start = time.time()
    print('Downloading dataset from %s' % URL)
    os.makedirs('data', exist_ok=True)

    r = requests.get(URL)
    if r.status_code != 200:
        print('Error: status_code=%d' % r.status_code)
        print(r.status_code)
    s = r.content.decode(ENCODING)
    lines = s.split('\n')

    flag = False # flag is True only in the range of urls that must be explored
    for line in lines:
        if not flag:
            if line.find('Composers') >= 0:
                flag = True
            continue
        if line.find('Genres') >= 0:
            break

        path = 'http://kern.humdrum.org/search?s=t&amp;keyword='
        source = re.search('keyword=(.+?)">', line)
        if source is None:
            continue

        name = source.group(1)
        print('Downloading scores from %s...' % name)
        os.makedirs(os.path.join('data', name), exist_ok=True)
        r2 = requests.get('%s%s' % (path, name)) # get source code of artist's page
        s2 = r2.content.decode(ENCODING)
        files = s2.split('\n')
        count = 0

        for file in files:
            file_found = re.search('location=users(.+?).krn&format=kern', file) # get file's url
            if file_found:
                count += 1
                file_name = 'data/%s/%02d.%s' % (name, count, EXTENSIONS[FORMAT])
                if os.path.isfile(file_name) and magic.from_file(file_name, mime=True) == 'audio/midi':
                    continue
                suffix = file_found.group(1).split('location=users')[-1]
                file_url = 'http://kern.humdrum.org/cgi-bin/ksdata?l=users%s.krn&f=%s' % (suffix, FORMAT)
                # writing the downloaded file on disk
                with open(file_name, 'wb') as f:
                    kern = requests.get(file_url)
                    f.write(kern.content)
                    if magic.from_file(file_name, mime=True) not in ['audio/midi','inode/x-empty']:
                        raise TypeError('Type of %s is %s' % (file_name, magic.from_file(file_name, mime=True)))

        print(' > Done, %d scores downloaded' % count)
    end = time.time()
    print('Scores successfully downloaded. Time elapsed: %ds' % (end-start))

if __name__ == '__main__':
    download()
