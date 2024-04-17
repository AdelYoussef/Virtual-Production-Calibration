import cv2
from VP_Calib import VP_Calib



def Live_Aruco_detect():
    Camera_Type = "VP"
    on = VP_Calib("config.yaml")
    Capture = on.Init_Live_Camera(Camera_Type , 0)


    on.Init_Aruco("/home/slam/SLAM/VP_Calib/A6400/Calibration_Data/A6400_Parameters.yaml")
    while True:
        Frame = on.Get_Frame(Camera_Type, Capture)
        
        if( on.Aruco_Detection(Frame) == True):
            print(on.Aruco_Trans_Vec)
        else:
            print("none")
        on.Crosshair(Frame,"Both")

        Frame = cv2.resize(Frame, on.VIEW_DIM)
        cv2.imshow("Frame", Frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Live_Aruco_detect()