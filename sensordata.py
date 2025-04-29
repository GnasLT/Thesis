from am2315 import AM2315
import adafruit_am2320
import board
import adafruit_bh1750 as ada
import mh_z19
import pymongo
import urllib
import datetime
import os
from pymongo import MongoClient
import socket

def read_co2():
    try:
        data = mh_z19.read_from_pwm()
        #data = data[data.find(':')+ 1: + data.find('}')]
        return data.get('co2')
    except:
        print("Sensor not found")
        return None

def read_light(i2c):
    try:
        data = ada.BH1750(i2c)
        return data
    except:
        print("Sensor not found")
        return None

def read_air(i2c):
    try:
        am = adafruit_am2320.AM2320(i2c)
        tem = am.temperature()
        humid = am.relative_humidity()
        #sens = AM2315.AM2315()
        #sens.read_humidity_temperature()
        return  tem, humid
    except:
        print("Sensor not found")
        return None

def readSensorData():
    time = datetime.datetime.now()
    i2c = board.I2C()
    co2 = read_co2()
    light = read_light(i2c)
    t,h  = read_air(i2c)
    saveSensorData('plant1',time,light,h,t,co2)

def getdb():
    user_name= 'gnas'
    pass_word = 'gnas'
    host = 'localhost'
    port =27017
    client = MongoClient(f'mongodb://{user_name}:{urllib.parse.quote_plus(pass_word)}@{host}:{port}')
    db = client['plant_data']
    return db
def saveSensorData(plantid,time,light,humid,tem,co2):
    db = getdb()
    collection = db['SensorData']
    data = {
        'plant_id': plantid,
        'time' : time,
        'values': [
            {
                'type': 'light',
                'value': light,
                'unit' : 'lux'
             },
            {
                'type': 'temperature',
                'value': tem,
                'unit' : '*C'
             },
            {
                'type': 'humidity',
                'value': humid,
                'unit' : '%RH'
             },
            {
                'type': 'co2',
                'value': co2,
                'unit' : 'ppm'
             }
            ]
        }
    insert_doc = collection.insert_one(data)

def saveImageData(plantid,rgbpath,nirpath,time):
    db = getdb()
    collection = db['ImageData']
    data = {
        'plant_id': plantid,
        'time' : time,
        'values': [
            {
                'type': 'rgb',
                'path': rgbpath,
                'resolution' : '2592x1944px'
             },
            {
                'type': 'nir',
                'path': nirpath,
                'resolution' : '3280x2464px'
             }
            ]
        }
    insert_doc = collection.insert_one(data)

def getImagefromSlave(time):
    image = f'rgb_{time}.png' #imagename = rgb_ + time
    IP = '192.168.1.23' #ip server
    port = 6666
    cmd = f'{image}' #command send to client
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #af_inet = ipv4, sockstream = TCP
    server_socket.bind((IP, port))
    server_socket.listen(1)
    print(f'Server is listening {IP}:{port}....')
    connection, client_address = server_socket.accept()# accept connection from client
    print(f'Connected to {client_address}')
    connection.sendall(cmd.encode('utf-8'))# send cmd to client
    print('Command sent')
    save_dir = os.path.join(os.path.expanduser('~'), 'Desktop/thesis')#get path to save images
    file = open(cmd,'wb') #get binary data from client and write a file
    chunk = connection.recv(1024 * 1024)# 1mb each time
    #write data received from client
    while chunk:
        file.write(chunk) #write data to file
        chunk = connection.recv(1024 * 1024)
    os.rename(f'{save_dir}/{image}',f'{save_dir}/images/{image}')#change image save path
    print('Received image')
    connection.close()
    server_socket.close()
    return f'{save_dir}/images/{image}'
def getImage(time):
    image_dir = os.path.join(os.path.expanduser('~'), 'Desktop/thesis/images')
    os.makedirs(image_dir, exist_ok=True)
    image = f'nir_{time}.png'
    image_path = os.path.join(image_dir, image)
    print (image_path)
    os.system(f'rpicam-still -e png -o {image_path}')
    return image_path


def CaptureAndSaveImage():
    time = datetime.datetime.now()
    time = time.strftime("%d-%m-%Y_%H:%M")
    rgbpath = getImagefromSlave(time)
    nirpath = getImage(time)
    saveImageData('plant1',rgbpath,nirpath,time)
