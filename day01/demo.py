# coding=utf-8
"""

"""
import urllib2
import urllib
import re
page = 1
url = 'http://www.qiushibaike.com/hot/page/' + str(page)
user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
headers = {'User-Agent' : user_agent }
try:
    request = urllib2.Request(url, headers=headers)
    response = urllib2.urlopen(request)
    content = response.read().decode('utf-8')
    # 找出对应的文本数据 以及评论数目
    pattern = re.compile('<div class="content">.*?<span>(.*?)</span>.*?<span class="stats-vote">.*?<i class="number">(.*?)</i>.*?</span>.*?<span class="stats-comments">.*?<i class="number">(.*?)</i>.*?</span>.*?</div>', re.S)
    items = re.findall(pattern, content)
    for item, number1, number2 in items:
        print item.replace("\n", "")  # 去除换行符号
        print "好笑数目:{0}".format(number1)
        print "评论数目:{0}".format(number2)
except urllib2.URLError, e:
    if hasattr(e,"code"):
        print e.code
    if hasattr(e,"reason"):
        print e.reason