# coding: utf-8
from BaseHTTPServer import BaseHTTPRequestHandler
from BaseHTTPServer import HTTPServer
from SocketServer import ThreadingMixIn
import threading
import cgi
from matplotlib import pyplot as plt
import cv2
import datetime
import random
import sys
import img_matcher

prev_img_base_filename = ""
prev_img_id = 0
img_filename_gen_lock = threading.Lock()

class MyHandler(BaseHTTPRequestHandler):

    html = """
<html>
<body>
<form action="/submit" method="post" enctype="multipart/form-data">
<p>SEARCH TARGET IMG <input type="file" name="img_target" accept="image/jpeg, image/png" /></p>
<p>QUERY IMG <input type="file" name="img_query" accept="image/jpeg, image/png" /></p>
<input type="submit" value="SEND" />
</form>
</body>
</html>
"""

    def extract_ext(self,filename):
        # 拡張子
        ext = ""
        
        if filename[-len(".png"):].lower() == ".png":
            ext = ".png"
        if filename[-len(".jpg"):].lower() == ".jpg":
            ext = ".jpg"

        return ext

    # 唯一のファイル名を生成する
    def gen_img_filename(self,ext):
        # make a filename
        img_base_filename = "./tmp/" + datetime.datetime.now().isoformat("-").replace(":","-").replace(".","-")
        
        global prev_img_base_filename
        global prev_img_id
        global img_filename_gen_lock

        with img_filename_gen_lock:

            if prev_img_base_filename == img_base_filename:
                img_id = prev_img_id + 1
                prev_img_id = img_id
            else:
                img_id = 0
                prev_img_id = img_id
                prev_img_base_filename = img_base_filename

        filename = img_base_filename + "_" + ("%04d"%img_id) + "_" \
        + ("".join([random.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for x in xrange(10)])) + ext

        return filename
 

    def save_img(self,field_item,ext):

        # The field contains an uploaded file
        file_data = field_item.file.read()
        file_len = len(file_data)

        filename = self.gen_img_filename(ext)
       
        # save
        fp = open(filename,"wb")
        fp.write(file_data)
        fp.close()

        return filename



    def do_POST(self):
        # POST されたフォームデータを解析する
        form = cgi.FieldStorage(
            fp=self.rfile, 
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })

        isOK = False
 
        if "img_target" in form and "img_query" in form:
            field_item_img_target = form["img_target"]
            field_item_img_query = form["img_query"]

            if field_item_img_target.filename and field_item_img_query.filename:

                ext_img_target = self.extract_ext(field_item_img_target.filename)
                ext_img_query = self.extract_ext(field_item_img_query.filename)

                if len(ext_img_target) > 0 and len(ext_img_query) > 0:

                    filename_img_target = self.save_img(field_item_img_target,ext_img_target)
                    filename_img_query = self.save_img(field_item_img_query,ext_img_query)
                    filename_img_result = self.gen_img_filename(".jpg")

                    # opencvで諸々の処理を実行
                    img_matcher.multi_query_matching( filename_img_query, filename_img_target, filename_img_result )

                    # 結果を返す
                    fp = open(filename_img_result,"rb")
                    self.send_response(200)
                    self.send_header('Content-type','image/jpg')
                    self.end_headers()
                    self.wfile.write(fp.read())
                    fp.close() 
                    
                    isOK = True

        if not isOK:

            self.send_response(200)
            self.end_headers()
            self.wfile.write(self.html)

        return

    def do_GET(self):

        # faviconのリクエストは無視
        if self.path.endswith('favicon.ico'):
            return;
        
        self.send_response(200)
        self.end_headers()
        
        self.wfile.write(self.html)


# if __name__ == '__main__':
#     server = HTTPServer(('localhost', 8080), MyHandler)
#     print 'Starting server, use <Ctrl-C> to stop'
#     server.serve_forever()


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

if __name__ == '__main__':
    server = ThreadedHTTPServer(('153.121.74.218', 80), MyHandler)
    print 'Starting server, use <Ctrl-C> to stop'
    server.serve_forever()

