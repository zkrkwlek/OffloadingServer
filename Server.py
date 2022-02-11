import threading
import ujson
import time
import numpy as np
from flask import Flask, request
import requests
import cv2
from socket import *
import argparse
import torch, torchvision
import os

#WSGI
from gevent.pywsgi import WSGIServer

#Thread Pool
from concurrent.futures import ThreadPoolExecutor

from module.ProcessingTime import ProcessingTime
import pickle
import keyboard

def printProcessingTime():
    Data["Time"].update()
    print(Data["Time"].print())
def saveProcessingTime():
    Data["Time"].update()
    pickle.dump(Data["Time"], open('./evaluation/processing_time.bin', "wb"))

#keyboard.add_hotkey("ctrl+p",lambda: printProcessingTime())
#keyboard.add_hotkey("ctrl+s",lambda: saveProcessingTime())

app = Flask(__name__)

def Display(res):
    for i, (im, pred) in enumerate(zip(res.imgs, res.pred)):
        n = len(pred)
        fdata = np.zeros(n*6, dtype=np.float32)  #n*6
        idx = 0
        for *box, conf, cls in reversed(pred):
            #label = res.names[int(cls)]
            #print("%s %f"%(label, conf))
            #print("%f %f %f %f"%(box[0], box[1], box[2], box[3]))
            fdata[idx] = float(cls)
            fdata[idx + 1] = conf
            fdata[idx + 2] = box[0]
            fdata[idx + 3] = box[1]
            fdata[idx + 4] = box[2]
            fdata[idx + 5] = box[3]
            idx = idx+6

        return fdata


def processingthread(message):
    print("Object Detection Start");
    t1 = time.time()
    global nConnect
    data = ujson.loads(message.decode())
    id = data['id']
    src = data['src']

    """
    #ConditionVariable.acquire()
    while len(models) < 1:
        abcabcabc = None
        #ConditionVariable.wait()

    mid,model = models.pop(0)
    """
    """
    ConditionVariable.acquire()
    while nConnect == 0:
        ConditionVariable.wait()
    nConnect = 0
    idx = -1
    while idx == -1:
        for i in range(0,10):
            if not uses[i]:
                idx = i
                uses[i] = True
                break
    nConnect = 1
    ConditionVariable.release()
    """
    res = sess.post(FACADE_SERVER_ADDR + "/Load?keyword=" + datatype + "&id=" + str(id) + "&src=" + src, "")
    img_array = np.frombuffer(res.content, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    #imgTensor = torch.from_numpy(img/255.).float()[3,None, None].to(device)
    #resized = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)



    ta = time.time()
    """
    im = torch.from_numpy(img).to(device).float()
    #im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        print(im.shape)
        im = im[None]  # expand for batch dim
    """
    with torch.no_grad():
        results = model(img)
    tb = time.time()
    fdata = Display(results)

    #models.append((mid,model))

    #ConditionVariable.notify()
    #ConditionVariable.release()
    t2 = time.time()

    """
    ConditionVariable.acquire()
    ConditionVariable.notify()
    uses[idx] = False
    ConditionVariable.release()
    """

    sess.post(FACADE_SERVER_ADDR + "/Store?keyword=ObjectDetection&id=" + str(id) + "&src=" + src, fdata.tobytes())

    mid = 1
    print("Object Detection = %s : %d, %f, %f" % (id, mid, t2 - t1, tb-ta))
    Data["Time"].add(t2 - t1, len(fdata))
    #print(id,t1,t2)

bufferSize = 1024
def udpthread():

    executor = ThreadPoolExecutor(4)

    while True:
        bytesAddressPair = ECHO_SOCKET.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        #with ThreadPoolExecutor() as executor:
        executor.submit(processingthread, message)

if __name__ == "__main__":

    ##################################################
    ##arguments
    parser = argparse.ArgumentParser(
        description='WISE UI Web Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--RKeywords', type=str,
        help='Received keyword lists')
    parser.add_argument(
        '--SKeywords', type=str,
        help='Sendeded keyword lists')
    parser.add_argument(
        '--DataType', type=str,
        help='Data type')
    parser.add_argument(
        '--ip', type=str,default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--port', type=int, default=35006,
        help='port number')
    parser.add_argument(
        '--use_gpu', type=str, default='0',
        help='port number')
    parser.add_argument(
        '--prior', type=str, default='0',
        help='port number')
    parser.add_argument(
        '--ratio', type=str, default='1',
        help='port number')
    parser.add_argument(
        '--FACADE_SERVER_ADDR', type=str,
        help='facade server address')
    parser.add_argument(
        '--PROCESS_SERVER_ADDR', type=str,
        help='process server address')
    parser.add_argument(
        '--ECHO_SERVER_IP', type=str, default='0.0.0.0',
        help='ip address')
    parser.add_argument(
        '--ECHO_SERVER_PORT', type=int, default=35001,
        help='port number')

    #load YOLO v5
    """
    ConditionVariable = threading.Condition()
    models = []
    nConnect = 1
    img_init = np.random.randint(255, size=(640, 480, 3), dtype=np.uint8)
    for i in range(0,10):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        model(img_init)
        models.append((i+1,model))
    executor = ThreadPoolExecutor(max_workers=len(models))
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('GPU')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained = True)
    model.eval().to(device)
    with torch.no_grad():
        rgb = np.random.randint(255, size=(640, 480, 3), dtype=np.uint8)
        model(rgb)
    Data = {}

    try:
        path = os.path.dirname(os.path.realpath(__file__))
        f = open(path + '/evaluation/processing_time.bin', 'rb')
        Data["Time"] = pickle.load(f)
        f.close()
    except FileNotFoundError:
        Data["Time"] = ProcessingTime()

    ##Echo server
    opt = parser.parse_args()
    FACADE_SERVER_ADDR = opt.FACADE_SERVER_ADDR
    ReceivedKeywords = opt.RKeywords.split(',')
    SendKeywords = opt.SKeywords
    datatype = opt.DataType

    sess = requests.Session()
    sess.post(FACADE_SERVER_ADDR + "/Connect", ujson.dumps({
        # 'port':opt.port,'key': keyword, 'prior':opt.prior, 'ratio':opt.ratio
        'id': 'YoloServer', 'type1': 'Server', 'type2': 'test', 'keyword': SendKeywords, 'Additional': None
    }))
    ECHO_SERVER_ADDR = (opt.ECHO_SERVER_IP, opt.ECHO_SERVER_PORT)
    ECHO_SOCKET = socket(AF_INET, SOCK_DGRAM)
    for keyword in ReceivedKeywords:
        temp = ujson.dumps({'type1': 'connect', 'keyword': keyword, 'src': 'FeatureServer', 'type2': 'all'})
        ECHO_SOCKET.sendto(temp.encode(), ECHO_SERVER_ADDR)
        Data[keyword] = {}
    # Echo server connect

    th1 = threading.Thread(target=udpthread)
    th1.start()

    http = WSGIServer((opt.ip, opt.port), app.wsgi_app)
    http.serve_forever()