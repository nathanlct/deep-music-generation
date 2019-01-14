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
        count = 1

        for file in files:
            file_found = re.search('<a href=(.+?).krn&format=kern', file) # get file's url
            if file_found:
                file_url = file_found.group(1)
                # writing the downloaded file on disk
                with open('data/%s/%02d.%s' % (name, count, EXTENSIONS[FORMAT]), 'wb') as f:
                    kern = requests.get('%s.krn&format=%s' % (file_url, FORMAT))
                    f.write(kern.content)
                    count += 1
        print(' > Done, %d scores downloaded' % (count-1))
    end = time.time()
    print('Scores successfully downloaded. Time elapsed: %ds' % (end-start))

if __name__ == '__main__':
    download()
