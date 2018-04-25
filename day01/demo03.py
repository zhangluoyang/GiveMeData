# coding=utf-8
"""
百度贴吧爬虫
"""
import re
import urllib
import urllib2

class BaiDu(object):

    # 初始化 传入基地址 是否只看楼主的参数
    def __init__(self, baseUrl, seeLZ):
        self.baseUrl = baseUrl
        self.seeLZ = '?see_lz='+str(seeLZ)

    # 传入页码，获取该帖子的代码
    def getPage(self, pageNum):
        try:
            url = self.baseUrl + self.seeLZ + '&pn=' + str(pageNum)
            request = urllib2.Request(url)
            response = urllib2.urlopen(request)
            return response.read().decode('utf-8')
        except urllib2.URLError, e:
            if hasattr(e, "reason"):
                print(u"连接百度贴吧失败，错误原因",e.reason)
                return None

    # 获取帖子的标题
    def getTitle(self):
        page = self.getPage(1)
        pattern = re.compile('<h.*?class="core_title_txt.*?">(.*?)</h.*?>', re.S)
        result = re.findall(pattern, page)
        if result:
            return result
        else:
            return None

    # 获取帖子的总数
    def getPageNum(self):
        page = self.getPage(1)
        pattern = re.compile('<li class="l_reply_num.*?</span>.*?<span.*?>(.*?)</span>', re.S)
        result = re.search(pattern, page)
        if result:
            # print result.group(1)  #测试输出
            return result.group(1).strip()
        else:
            return None

    # 获取具体数据内容
    def getContent(self, page):
        pattern = re.compile('<div id="post_content_.*?>(.*?)</div>', re.S)
        items = re.findall(pattern, page)
        for item in items:
            print(item)
if __name__ == "__main__":
    baseURL = 'http://tieba.baidu.com/p/3138733512'
    bdtb = BaiDu(baseURL, 1)
    page = bdtb.getPage(1)
    bdtb.getContent(page)