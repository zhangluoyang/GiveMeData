# coding=utf-8
"""
python网络爬虫
"""
import urllib
import urllib2
import re
import thread
import time


# 糗事百科爬虫类
class QSBK(object):

    def __init__(self):
        # 糗事百科的前n个top页面
        self.pageIndex = 1
        self.user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
        # 初始化headers
        self.headers = {'User-Agent': self.user_agent}
        # 存放段子的变量，每一个元素是每一页的段子们
        self.stories = []
        # 存放程序是否继续运行的变量
        self.enable = False

    # 传入某一页的索引获得页面代码
    def getPage(self, pageIndex):
        try:
            url = 'http://www.qiushibaike.com/hot/page/' + str(pageIndex)
            # 构建请求的request
            request = urllib2.Request(url, headers=self.headers)
            # 利用urlopen获取页面代码
            response = urllib2.urlopen(request)
            # 将页面转化为UTF-8编码
            pageCode = response.read().decode('utf-8')
            return pageCode

        except urllib2.URLError, e:
            if hasattr(e, "reason"):
                print u"连接糗事百科失败,错误原因", e.reason
                return None

    # 获取具体某一页面的内容
    def getPageItems(self, pageIndex):
        pageCode = self.getPage(pageIndex)
        if not pageCode:
            print "页面加载失败...."
            return None
        pattern = re.compile('<div class="content">.*?<span>(.*?)</span>.*?<span class="stats-vote">.*?<i class="number">(.*?)</i>.*?</span>.*?<span class="stats-comments">.*?<i class="number">(.*?)</i>.*?</span>.*?</div>',re.S)
        items = re.findall(pattern, pageCode)
        pageStories = []
        for item, number1, number2 in items:
            pageStories.append([item, number1, number2])
        return pageStories

    # 加载并提取页面的内容，加入到列表中
    def loadPage(self):
        # 如果当前未看的页数少于2页，则加载新一页
        if self.enable == True:
            if len(self.stories) < 2:
                # 获取新一页
                pageStories = self.getPageItems(self.pageIndex)
            # 将该页的段子存放到全局list中
                if pageStories:
                    self.stories.append(pageStories)
                    # 获取完之后页码索引加一，表示下次读取下一页
                    self.pageIndex += 1





if __name__ == "__main__":
    pass