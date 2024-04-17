import asyncio
import websockets
import cv2
from cv2 import aruco
import numpy as np
import json
import time
from VP_Calib import VP_Calib




async def server(websocket,path):

    Camera_Type = "VP"
    on = VP_Calib("config.yaml")
    Capture = on.Init_Live_Camera(Camera_Type , 0)
    on.Init_Aruco("/home/slam/SLAM/VP_Calib/A6400/Calibration_Data/A6400_Parameters.yaml")

    while True:
        Frame = on.Get_Frame(Camera_Type, Capture)
        if( on.Aruco_Detection(Frame) == True):
            msg = json.dumps({
                    'x':str((on.Inv_Aruco_Trans_Vec[0])),
                    'y':str((on.Inv_Aruco_Trans_Vec[1])),
                    'z':str((on.Inv_Aruco_Trans_Vec[2])),
                    'deg_x':str(float(on.Inv_Euler_Angles_Aruco[0])),
                    'deg_y':str(float(on.Inv_Euler_Angles_Aruco[1])),
                    'deg_z':str(float(on.Inv_Euler_Angles_Aruco[2])),
                    })
            
            await websocket.send(msg)  
            await asyncio.sleep(0.01) 
            print("message sent")
        else:
            print("none")

        on.Crosshair(Frame)

        Frame = cv2.resize(Frame, on.VIEW_DIM)
        cv2.imshow("Frame", Frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            Stop_Server()
            cv2.destroyAllWindows()
            break


def Start_Server(address = "localhost", port = 6969):
    print("server will start")
    Server = websockets.serve(server, address, port)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(Server)
    print("waiting for unreal to connet")
    loop.run_forever()
    

def Stop_Server():
    asyncio.get_event_loop().stop()
    print("server terminated")


Start_Server("192.168.1.11")