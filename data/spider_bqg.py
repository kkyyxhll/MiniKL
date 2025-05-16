import requests
import urllib3
import os
import json
import time
import re
import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm


class SpiderBQG:
    def __init__(self, book_start_num, book_end_num, data_jsonl_path=''):
        self.data_jsonl_path = self._init_data_jsonl_path(data_jsonl_path)
        self.book_start_num = book_start_num
        self.book_end_num = book_end_num

        # 随机User-Agent池，避免单一UA被封禁
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 11.5; rv:90.0) Gecko/20100101 Firefox/90.0"
        ]

        self.headers = {
            'Connection': 'close',
        }
        self.id_content_dict = {}
        self.lock = Lock()  # 用于线程安全
        self.session = requests.Session()  # 复用会话

    def spider(self):
        # 禁用不安全请求警告
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        for book_seq in range(self.book_start_num, self.book_end_num + 1):
            start_time = time.time()

            try:
                # 获取章节数量
                chapter_num = self._get_chapter_num_by_book_seq(book_seq)
                print(f"开始爬取书籍 {book_seq}，共 {chapter_num} 章")

                # 多线程爬取章节内容
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_chapter = {
                        executor.submit(self._get_content_by_bs_and_cs, book_seq, chapter_seq): chapter_seq
                        for chapter_seq in range(1, chapter_num + 1)
                    }

                    # 使用tqdm显示进度条
                    for future in tqdm(as_completed(future_to_chapter), total=chapter_num,
                                       desc=f"书籍 {book_seq} 进度"):
                        chapter_seq = future_to_chapter[future]
                        try:
                            future.result()  # 获取结果，触发异常处理
                        except Exception as e:
                            print(f"章节 {chapter_seq} 爬取失败: {str(e)}")

                # 写入数据
                self._write_data_jsonl(book_seq)
                print(f"书籍 {book_seq} 爬取完成，耗时: {time.time() - start_time:.2f}秒")

            except Exception as e:
                print(f"书籍 {book_seq} 爬取异常: {str(e)}")
                continue

    def _get_chapter_num_by_book_seq(self, book_seq):
        """获取书籍的章节数量"""
        base_url = "https://www.bie5.cc/html/"
        url = base_url + str(book_seq)

        try:
            # 设置随机User-Agent
            self.headers['User-Agent'] = random.choice(self.user_agents)
            resp = self.session.get(url, headers=self.headers, verify=False, timeout=10)
            time.sleep(random.uniform(0.5, 1))  # 随机延时
            resp.encoding = 'utf-8'

            text = resp.text
            # 提取章节数量
            pattern = f'<a href ="/html/{book_seq}/(.*?).html">'
            data = re.findall(pattern, text, re.DOTALL)
            return len(data)

        except Exception as e:
            print(f"获取书籍 {book_seq} 章节数量失败: {str(e)}")
            raise

    def _get_content_by_bs_and_cs(self, book_seq, chapter_seq):
        """获取指定书籍和章节的内容"""
        id = f"{book_seq}|{chapter_seq}"
        base_url = "https://www.bie5.cc/html/"
        url = base_url + str(book_seq) + "/" + str(chapter_seq) + ".html"

        try:
            # 设置随机User-Agent
            self.headers['User-Agent'] = random.choice(self.user_agents)
            resp = self.session.get(url, headers=self.headers, verify=False, timeout=10)
            resp.encoding = 'utf-8'

            text = resp.text
            # 提取内容
            pattern = r'<div id="chaptercontent" class="Readarea ReadAjax_content">(.*?)请收藏本站'
            data = re.findall(pattern, text, re.DOTALL)[0]
            # 清理内容
            data = re.sub(r'\s+', '', data)
            data = data.replace('<br/><br/>', '\n')

            # 线程安全地添加到字典
            with self.lock:
                self.id_content_dict[id] = data

            # 随机延时，避免请求过于频繁
            time.sleep(random.uniform(0.3, 0.8))

        except Exception as e:
            print(f"获取章节 {chapter_seq} 内容失败: {str(e)}")
            raise

    def _write_data_jsonl(self, book_seq):
        """将爬取的数据写入JSONL文件"""
        try:
            # 线程安全地获取数据并清空
            with self.lock:
                book_data = {
                    id: content
                    for id, content in self.id_content_dict.items()
                    if id.startswith(f"{book_seq}|")
                }
                # 从字典中删除已写入的数据
                for id in book_data.keys():
                    del self.id_content_dict[id]

            # 写入文件
            if book_data:
                with open(self.data_jsonl_path, "a", encoding="utf-8") as f:
                    for id, content in book_data.items():
                        temp_dict = {"id": id, "content": content}
                        json.dump(temp_dict, f, ensure_ascii=False)
                        f.write("\n")
                print(f"书籍 {book_seq} 数据已写入: {self.data_jsonl_path}")
            else:
                print(f"书籍 {book_seq} 没有数据可写入")

        except Exception as e:
            print(f"写入书籍 {book_seq} 数据失败: {str(e)}")

    @staticmethod
    def _init_data_jsonl_path(data_jsonl_path):
        """初始化数据文件路径"""
        if data_jsonl_path == "":
            temp_data_jsonl_path = os.path.join(os.path.dirname(__file__), "data.jsonl")
        else:
            if os.path.isabs(data_jsonl_path):
                temp_data_jsonl_path = data_jsonl_path
            else:
                temp_data_jsonl_path = os.path.join(os.path.dirname(__file__), data_jsonl_path)

        os.makedirs(os.path.dirname(temp_data_jsonl_path), exist_ok=True)
        print(f"数据将保存到: {temp_data_jsonl_path}")
        return temp_data_jsonl_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多线程爬取笔趣阁小说')
    parser.add_argument('--book_start_num', type=int, default=201, help='开始书籍编号')
    parser.add_argument('--book_end_num', type=int, default=300, help='结束书籍编号')
    parser.add_argument('--data_jsonl_path', type=str, default='', help='数据保存路径')
    args = parser.parse_args()

    bqg_spider = SpiderBQG(args.book_start_num, args.book_end_num, args.data_jsonl_path)
    bqg_spider.spider()