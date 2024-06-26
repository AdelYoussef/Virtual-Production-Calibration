import os
import cv2
import yaml
import glob
import yaml
import numpy as np
from cv2 import aruco
import pyzed.sl as sl
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation   


class VP_Calib:

    # Class Constructor that gets called eveytime a new object is created from the class
    #sets basic parameters and calls Load_Config function that sets environment variables
    def __init__(self,config_file = "None"):
        
        np.set_printoptions(suppress = True)
        self.current_directory = os.path.dirname(os.path.realpath(__file__))

        self.Load_Config(config_file)

        self.ZED_MAT = sl.Mat()
        self.ZED_runtime_parameters = sl.RuntimeParameters()
        
        self.mono_calibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.stereo_calibration_criteria = (cv2.TERM_CRITERIA_MAX_ITER +cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        print(self.current_directory)

    # reads the config.yaml that hold current values of environment variables 
    def Load_Config(self,config_file):
        config_file_path = os.path.join(self.current_directory , config_file)

        with open(config_file_path, 'r') as stream:
            config = yaml.safe_load(stream)

        self.VP_Cam_sensor_width =  config["VP_Cam_sensor_width"]
        self.VP_Cam_sensor_height = config["VP_Cam_sensor_height"]
        self.VP_Cam_pixels_horizontal = config["VP_Cam_pixels_horizontal"]
        self.VP_Cam_pixels_vertical = config["VP_Cam_pixels_vertical"]

        self.CHECKERBOARD_SIZE = config["CHECKERBOARD_SIZE"]
        self.ARUCO_MARKER_SIZE = config["ARUCO_MARKER_SIZE"]
        self.ARUCO_MARKER_ID = config["ARUCO_MARKER_ID"]
        self.ARUCO_MODE = config["ARUCO_MODE"]
        FRAME_WIDTH = config["FRAME_WIDTH"]
        FRAME_HEIGHT = config["FRAME_HEIGHT"]
        VIEW_RESIZE = config["VIEW_RESIZE"]
        
        CHECKERBOARD_ROWS = config["CHECKERBOARD_ROWS"]
        CHECKERBOARD_COLUMNS = config["CHECKERBOARD_COLUMNS"]
        
        self.FRAME_DIM = (FRAME_WIDTH , FRAME_HEIGHT) 
        self.CHESS_BOARD_DIM = (CHECKERBOARD_ROWS , CHECKERBOARD_COLUMNS)
        self.VIEW_DIM = (int(FRAME_WIDTH * VIEW_RESIZE) , int(FRAME_HEIGHT * VIEW_RESIZE)) 
        
        self.show_Calibration_images = config["show_Calibration_images"]
        
    #inits the live camera capturer based on the passed type of the camera
    def Init_Live_Camera(self, Camera_Type = None, id = None):

        #normal webcams usually run by 720x480 
        #if the webcam supports 1920x1080 then we need to command it to operate at 1920x1080
        if (Camera_Type == "WEBCAM1080"):
            webcam1080 = cv2.VideoCapture(id)
            webcam1080.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            webcam1080.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            webcam1080.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            return webcam1080
        
        #in case of a virtual production camera "camlink" or a mormal webcam mode , we care for the default resolution
        if (Camera_Type == "WEBCAM" or Camera_Type == "VP"):
            camera = cv2.VideoCapture(id)
            return camera

        #init ZED2 Camera using ZED SDK APi
        if (Camera_Type == "ZED") :
            zed = sl.Camera()
             # Create a InitParameters object and set configuration parameters
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD1080 # Use HD720 opr HD1200 video mode, depending on camera type.
            init_params.camera_fps = 30  # Set fps at 30
            # Open the camera
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print("Camera Open : "+repr(err)+". Exit program.")
                exit()
            zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 3)
            zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)
            zed.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 0)
            zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)
            zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 5)
            zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 74)
            zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 100)
            zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
            return zed

        else:
            print("error provided camera class isn't recognized/n")
            print("please provide one of the following/n")
            print("ZED/n")
            print("WEBCAM/n")
            print("WEBCAM1080/n")
            print("VP/n")
                

    #grabs a frame from the operating image based on the passed category and the passed id
    def Get_Frame(self, Camera_Type, Capture):
        if (Camera_Type == "WEBCAM" or Camera_Type == "VP" or Camera_Type == "WEBCAM1080"):
            _, Frame = Capture.read()
            return Frame

        if (Camera_Type == "ZED") :
           if Capture.grab(self.ZED_runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            Capture.retrieve_image(self.ZED_MAT, sl.VIEW.LEFT)
            Frame = self.ZED_MAT.get_data()
            return Frame

        else:
            print("error provided camera class isn't recognized/n")
            print("please provide one of the following/n")
            print("ZED/n")
            print("WEBCAM/n")
            print("WEBCAM1080/n")
            print("VP/n")
            return False      
            

    # just a dummy crosshair that marks the center of the image, can be very useful
    def Crosshair(self,Frame,condition = "None"):
        cv2.putText(Frame, f"crosshair mode---> {condition}", (1400, 50), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0, 255, 0) , 2, cv2.LINE_AA) 
        if condition == "Ideal":
            width, height = Frame.shape[1] , Frame.shape[0]
            cv2.line(Frame, (int(width/2), 0) , (int(width/2), height) , (0, 255, 0), 2)  #vertical line
            cv2.line(Frame, (0, int(height/2)) , (width, int(height/2)) , (0, 255, 0), 2) #horizontal line
        
        if condition == "Calibrated":
            cv2.line(Frame, (self.Crosshair_Cx, 0) , (self.Crosshair_Cx, height) , (0, 0, 255), 2)  #vertical line
            cv2.line(Frame, (0, self.Crosshair_Cy) , (width, self.Crosshair_Cy) , (0, 0, 255), 2)   #horizontal line

        if condition == "Both":
            width, height = Frame.shape[1] , Frame.shape[0]
            cv2.line(Frame, (int(width/2), 0) , (int(width/2), height) , (0, 255, 0), 2)  #vertical line
            cv2.line(Frame, (0, int(height/2)) , (width, int(height/2)) , (0, 255, 0), 2) #horizontal line

            cv2.line(Frame, (self.Crosshair_Cx, 0) , (self.Crosshair_Cx, height) , (0, 0, 255), 2)  #vertical line
            cv2.line(Frame, (0, self.Crosshair_Cy) , (width, self.Crosshair_Cy) , (0, 0, 255), 2)   #horizontal line
            
        return Frame

    # inits the detection and pose estimation of an aruco marker, sets aruco dict, inits the transformation matrix, as well as loads the used camera matrix and distortion parameters
    def Init_Aruco(self,Camera_Parameters_File_Path):
        self.Aruco_Cam_Mat, self.Aruco_Cam_Dist = self.Load_Cam_Parameters(Camera_Parameters_File_Path)
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
        self.Camera_Transform_Aruco = np.eye(4, dtype=float)
        
    #detects the presence of an aruco marker in a passed frame and returns a new frame with plotted detection
    def Aruco_Detection(self, Frame):
        Detection_State = False
        self.Camera_marker_corners,  self.Camera_marker_IDs, Camera_reject = aruco.detectMarkers(Frame, self.marker_dict)
        if self.Camera_marker_corners:
            Camera_rVec_Aruco, Camera_tVec_Aruco, _ = aruco.estimatePoseSingleMarkers(self.Camera_marker_corners, self.ARUCO_MARKER_SIZE, self.Aruco_Cam_Mat, self.Aruco_Cam_Dist)
            index = np.where( self.Camera_marker_IDs==self.ARUCO_MARKER_ID)[0][0]

            self.Aruco_corners  = self.Camera_marker_corners[index]
            self.Aruco_Rot_Vec = np.squeeze(Camera_rVec_Aruco[index])
            self.Aruco_Trans_Vec = np.squeeze(Camera_tVec_Aruco[index])


            self.Camera_Transform_Aruco[:3,3] = self.Aruco_Trans_Vec
            self.Camera_Transform_Aruco[:3,:3] , _ = cv2.Rodrigues(self.Aruco_Rot_Vec)
            Rot =  Rotation.from_matrix(self.Camera_Transform_Aruco[:3,:3] )
            self.Euler_Angles_Aruco = Rot.as_euler("xyz",degrees=True)
            
            self.Aruco_Transform_Camera = np.linalg.inv(self.Camera_Transform_Aruco)

            self.Inv_Aruco_Trans_Vec = self.Aruco_Transform_Camera [:3,3]
            Inv_Rot = Rotation.from_matrix(self.Aruco_Transform_Camera[:3,:3])
            self.Inv_Euler_Angles_Aruco = Inv_Rot.as_euler("xyz",degrees=True)

            self.Arcuo_Draw_Detection(Frame)
            Detection_State = True

        return(Detection_State)

                

        
    #draws the plotted detection of any aruco marker present in the camera frame
    def Arcuo_Draw_Detection(self, Frame):
        corners = self.Aruco_corners.reshape(4, 2)
        top_left = corners[0].ravel().astype(int)
        top_right = corners[1].ravel().astype(int)
        bottom_right = corners[2].ravel().astype(int)
        bottom_left = corners[3].ravel().astype(int)

        cv2.polylines(Frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv2.LINE_AA )
        cv2.circle(Frame, top_left, 2, (0,0,0), 2)
        cv2.circle(Frame, top_right, 2, (0,0,255), 2)
        cv2.circle(Frame, bottom_right, 2, (255,0,255), 2)
        cv2.circle(Frame, bottom_left, 2, (0,255,0), 2)

        cv2.drawFrameAxes(Frame, self.Aruco_Cam_Mat, self.Aruco_Cam_Dist, self.Aruco_Rot_Vec, self.Aruco_Trans_Vec, 100, 5)

        if(self.ARUCO_MODE == "Aruco Pose"):
            cv2.putText(Frame,
                        f"x:{round(self.Aruco_Trans_Vec[0],1)} y: {round(self.Aruco_Trans_Vec[1],1)} z: {round(self.Aruco_Trans_Vec[2],1)} ",
                        (50, 50),cv2.FONT_HERSHEY_PLAIN,1.5,(0, 0, 255),4,cv2.LINE_AA,)

            cv2.putText(Frame,
                        f"Rot x:{round(self.Euler_Angles_Aruco[0],1)} Rot y: {round(self.Euler_Angles_Aruco[1],1)} Rot z: {round(self.Euler_Angles_Aruco[2],1)} "  ,
                        (75, 75),cv2.FONT_HERSHEY_PLAIN,1.5,(0, 0, 255),4,cv2.LINE_AA,)


        if(self.ARUCO_MODE == "Camera Pose"):
            cv2.putText(Frame,
                        f"x:{round(self.Inv_Aruco_Trans_Vec[0],1)} y: {round(self.Inv_Aruco_Trans_Vec[1],1)} z: {round(self.Inv_Aruco_Trans_Vec[2],1)} ",
                        (50, 50),cv2.FONT_HERSHEY_PLAIN,1.5,(0, 0, 255),4,cv2.LINE_AA,)

            cv2.putText(Frame,
                        f"Rot x:{round(self.Inv_Euler_Angles_Aruco[0],1)} Rot y: {round(self.Inv_Euler_Angles_Aruco[1],1)} Rot z: {round(self.Inv_Euler_Angles_Aruco[2],1)} "  ,
                        (75, 75),cv2.FONT_HERSHEY_PLAIN,1.5,(0, 0, 255),4,cv2.LINE_AA,)
            


    #starts a live feed of any camera passed for the purpose of saving a single image to be used later, or just simple monitor for the feed
    def Capture_Single_Image(self, Camera_Type, id = 0, Camera_Name = "None"):
        Capture = self.Init_Live_Camera(Camera_Type , id)
        save_path = f"{self.current_directory}/{Camera_Name}.png"
        while True:
            Frame = self.Get_Frame(Camera_Type, Capture)
            Frame = cv2.resize(Frame, self.VIEW_DIM)
            cv2.imshow("Frame", Frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
            if key == ord("s"):
                cv2.imwrite(save_path, Frame)
                cv2.putText(Frame, f"image saved---> {Camera_Name}.png", (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0, 255, 0) , 2, cv2.LINE_AA) 
        return save_path

    #starts a live feed with purpose of capturing and saving images to be used in camera calibration later 
    #caputred images need to have a detected checkerboard after setting it's rows and columbns in the config file
    def Mono_Calibration_Capture(self, Camera_Type = "None", id = 0, Camera_Name = "None"):
        Capture = self.Init_Live_Camera(Camera_Type , id)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")     
        save_folder = f"{self.current_directory}/{Camera_Name}/Images/{dt_string}"
        os.makedirs(save_folder)
        print(f'save folder created ---> {save_folder}')   
        save_count = 0
        while True:
            Frame = self.Get_Frame(Camera_Type, Capture)
            Save_Frame = Frame.copy()
            grayScale = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
            board_detected, corners = cv2.findChessboardCorners(Frame, self.CHESS_BOARD_DIM, None)
            if board_detected == True:
                corners_draw = cv2.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), self.mono_calibration_criteria)
                cv2.drawChessboardCorners(Frame, self.CHESS_BOARD_DIM, corners_draw, board_detected)                
                
            cv2.putText(Frame, str(save_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0, 255, 0) , 2, cv2.LINE_AA) 
            Frame = cv2.resize(Frame, self.VIEW_DIM)
            cv2.imshow("Live Frame mono calib", Frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break
            if key == ord("s"):
                cv2.imwrite(f"{save_folder}/image{save_count}.png", Save_Frame)
                save_count +=1

        print("images saved in file ---> " + save_folder)
        return (save_folder)

    #starts a live feed with purpose of capturing and saving images to be used in stereo pair calibration later 
    #caputred images need to have a detected checkerboard after setting it's rows and columbns in the config file
    #operates using two cameras at the same time  
    def Stereo_Calibration_Capture(self, Camera_Type_VP, Camera_Type_TR, id_1, id_2, Camera_Name_VP, Camera_Name_TR):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S") 
        TR_Capture = self.Init_Live_Camera(Camera_Type_VP , id_1)
        VP_Capture = self.Init_Live_Camera(Camera_Type_TR , id_2)
        save_count = 0
        while True:
            VP_Frame = self.Get_Frame(Camera_Type_VP, VP_Capture)
            TR_Frame = self.Get_Frame(Camera_Type_TR, TR_Capture)

            TR_grayScale = cv2.cvtColor(TR_Frame, cv2.COLOR_BGR2GRAY)
            VP_grayScale = cv2.cvtColor(VP_Frame, cv2.COLOR_BGR2GRAY)
            
            TR_board_detected, corners = cv2.findChessboardCorners(TR_grayScale, self.CHESS_BOARD_DIM, None)
            VP_board_detected, corners = cv2.findChessboardCorners(VP_grayScale, self.CHESS_BOARD_DIM, None)

            if VP_board_detected == True and TR_board_detected == True:
                VP_corners_draw = cv2.cornerSubPix(TR_grayScale, corners, (3, 3), (-1, -1), self.mono_calibration_criteria)# criteria needs editing
                TR_corners_draw = cv2.cornerSubPix(VP_grayScale, corners, (3, 3), (-1, -1), self.mono_calibration_criteria)# criteria needs editing

                cv2.drawChessboardCorners(VP_Frame, self.CHESS_BOARD_DIM, VP_corners_draw, VP_board_detected)
                cv2.drawChessboardCorners(TR_Frame, self.CHESS_BOARD_DIM, TR_corners_draw, TR_board_detected)

            if(self.show_images):
                VP_Frame = cv2.resize(VP_Frame, self.VIEW_DIM)
                TR_Frame = cv2.resize(TR_Frame, self.VIEW_DIM)
                cv2.imshow("Live Frame Virtual Production", VP_Frame)
                cv2.imshow("Live Frame Tracker", TR_Frame)
            
            key = cv2.waitKey(1)
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

            if key == ord("s"):
                cv2.imwrite(f"{Camera_Name_VP}/Stereo_Images/{dt_string}/image{save_count}.png", VP_Frame)
                cv2.imwrite(f"{Camera_Name_TR}/Stereo_Images/{dt_string}/{save_count}.png", TR_Frame)
                save_count +=1

    #sets the holder list of the detect 2D checkboard corners as well as the corresponding 3D points
    def Init_CheckerBoard(self,):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        obj_3D = np.zeros((self.CHESS_BOARD_DIM[0] * self.CHESS_BOARD_DIM[1], 3), np.float32)
        obj_3D[:, :2] = np.mgrid[0 : self.CHESS_BOARD_DIM[0], 0 : self.CHESS_BOARD_DIM[1]].T.reshape(-1, 2)
        obj_3D *= self.CHECKERBOARD_SIZE
        
        # Arrays to store object points and image points from all the images.
        obj_points_3D = []  # 3d point in real world space
        img_points_2D = []  # 2d points in image plane.
        print("3D points ready , initialization is done")

        return True , obj_3D , obj_points_3D , img_points_2D
    
    #loads images in a folder, from a passed folder path and returns a list of the loaded images
    def Load_Images(self,image_folder):
        print("images will be loaded ")
        images = []
        images_names = sorted(os.listdir(image_folder))
        iter_list = tqdm(images_names)
        for image_name in iter_list:
            iter_list.set_postfix({'loading --> ': image_name})
            imagePath = os.path.join(image_folder , image_name)
            image = cv2.imread(imagePath)
            images.append(image)

        (Image_Height , Image_Width) = images[0].shape[:2]
        print((Image_Width, Image_Height))
        print("loading done")
        return images, (Image_Width, Image_Height) 
     
    # checks for a checkerboard present in an image, and returns list of 2D detected corners and the corresponding 3D points
    def Find_CheckerBoard_Corners(self,images):
        
        Init_State , obj_3D , obj_points_3D , img_points_2D = self.Init_CheckerBoard()
        if(Init_State and images):
            print("staring checkerboard detection")

            iter_list = tqdm(images)
            for index , image in enumerate(iter_list):
                iter_list.set_postfix({'checking image --> ': str(index)})
                grayScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(image, self.CHESS_BOARD_DIM, None)
                if ret == True:
                    obj_points_3D.append(obj_3D)
                    corners = cv2.cornerSubPix(grayScale, corners, (3, 3), (-1, -1), self.mono_calibration_criteria)
                    img_points_2D.append(corners)
                    image = cv2.drawChessboardCorners(image, self.CHESS_BOARD_DIM, corners, ret)

                if(self.show_Calibration_images):
                    cv2.imshow(("image" + str(index)), image)
                    cv2.waitKey(500)
                    cv2.destroyWindow(("image" + str(index)))

        print("checkerboard detection done and points extracted")
        return True , img_points_2D , obj_points_3D
    
    # mono calibration function that calculates the camera matrix and distortion coeffs of an arbitrary camera 
    def Mono_Calib(self,images_folder):
        parent_folder = os.path.dirname(os.path.dirname(images_folder))
        images, FRAME_DIM = self.Load_Images(images_folder)
        Detection_State , img_points_2D , obj_points_3D = self.Find_CheckerBoard_Corners(images)

        if(Detection_State):
            print("staring calibration")
            RMS_error, Cam_Mat, Cam_dist, Cam_rvecs, Cam_tvecs = cv2.calibrateCamera(obj_points_3D, img_points_2D,FRAME_DIM, None, None)      
        print("calibration done")
        mean_error = self.Mean_Calib_Error(obj_points_3D, img_points_2D, Cam_rvecs, Cam_tvecs, Cam_Mat, Cam_dist)
        FOV_Camera_Data = self.VP_Camera_Parameters(Cam_Mat)

        self.Save_Mono_Calib(parent_folder, FRAME_DIM, Cam_Mat, Cam_dist, RMS_error, mean_error, FOV_Camera_Data)

        return True

    # inits the stereo calibration folder by laoding calibration images + calculated camera parameters of the camera pair
    def Init_Stereo_Calib(self, Parent_Folder):
        images_folder  = os.path.join(Parent_Folder,"stereo_images")

        Directory , Camera_Name = os.path.split(Parent_Folder)
        cam_parameters_file  = os.path.join(Parent_Folder,f"Calibration_Data/{Camera_Name}_Parameters.yaml")
        images, FRAME_DIM = self.Load_Images(images_folder)
        Detection_State , img_points_2D , obj_points_3D = self.Find_CheckerBoard_Corners(images)
        MTX , DIST = self.Load_Cam_Parameters(cam_parameters_file)

        return Detection_State, FRAME_DIM, img_points_2D, obj_points_3D, MTX, DIST, Camera_Name
    

    # runs stereo calibration to retrive the relative pose between two cameras
    def Stereo_Calib(self, Parent_Folder_VP, Parent_Folder_TR):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_FIX_K1
        flags |= cv2.CALIB_FIX_K2
        flags |= cv2.CALIB_FIX_K3
        flags |= cv2.CALIB_FIX_K4
        flags |= cv2.CALIB_FIX_K5
        flags |= cv2.CALIB_FIX_K6
        #flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        #flags |= cv2.CALIB_FIX_ASPECT_RATIO
        #flags |= cv2.CALIB_ZERO_TANGENT_DIST

        Detection_State_VP , FRAME_DIM_VP , img_points_2D_VP , obj_points_3D_VP, MTX_VP , DIST_VP, Camera_Name_VP =  self.Init_Stereo_Calib(Parent_Folder_VP)

        Detection_State_TR , FRAME_DIM_VP , img_points_2D_TR , obj_points_3D_TR, MTX_TR , DIST_TR, Camera_Name_TR =  self.Init_Stereo_Calib(Parent_Folder_TR)


        if(Detection_State_VP == True and  Detection_State_TR == True):
            print("staring stereo calibration")
            RMS_error, MTX_VP, DIST_VP, MTX_TR, DIST_TR, R, T, E, F = cv2.stereoCalibrate(obj_points_3D_VP,
                                                                               img_points_2D_VP,
                                                                                 img_points_2D_TR, 
                                                                                 MTX_VP, 
                                                                                 DIST_VP, 
                                                                                 MTX_TR,
                                                                                 DIST_TR, 
                                                                                 FRAME_DIM_VP,
                                                                                 criteria=self.stereo_calibration_criteria, 
                                                                                 flags=flags)

        r =  Rotation.from_matrix(R)
        angles = r.as_euler("xyz",degrees=True)

        print("stereo calibration done") 
        print("relative pose ----->")       
        print("RMS ----->" + str(RMS_error))       
        print("relative translation ----->" +str(T))       
        print("relative rotation ----->" + str(angles))    

        Save_File = os.path.join(self.current_directory, f"{Camera_Name_VP}__{Camera_Name_TR}.yaml")
        self.Save_Stereo_Calib(RMS_error, T, angles, Save_File)
        return True
    
    # calculates the mean error of camera calibration process
    def Mean_Calib_Error(self,obj_points_3D, img_points_2D, Cam_rvecs, Cam_tvecs, Cam_Mat, Cam_dist):
        print("Calculating Mean Error ")
        total_error = 0
        for i in range(len(obj_points_3D)):
            imgpoints2, _ = cv2.projectPoints(obj_points_3D[i], Cam_rvecs[i], Cam_tvecs[i], Cam_Mat, Cam_dist)
            error = cv2.norm(img_points_2D[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            total_error += error
        mean_error = total_error/len(obj_points_3D)
        return mean_error
    
    # loads camera matrix and distortion coeffs to be used by any code snippet
    def Load_Cam_Parameters(self, filepath):
        with open(filepath, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        image_width = data_loaded["Frame Width"]
        image_height = data_loaded["Frame Height"]
        fx = data_loaded["fx"]
        fy = data_loaded["fy"]
        cx = data_loaded["cx"]
        cy = data_loaded["cy"]
        k1 = data_loaded["k1"]
        k2 = data_loaded["k2"]
        k3 = data_loaded["k3"]
        p1 = data_loaded["p1"]
        p2 = data_loaded["p2"]
        self.Crosshair_Cx = int(round(cx))
        self.Crosshair_Cy = int(round(cy))
        Cam_Dist =  np.array([k1 , k2 , k3 , p1 , p2])
        Cam_Mat = np.array([[fx , 0 , cx],[0 , fy , cy],[0 , 0 , 1]])
        print(f"Loading done  --->{filepath}")
        return Cam_Mat , Cam_Dist
        
    # saves calibrated camera matrix and distortion coeffs 
    def Save_Mono_Calib(self, Parent_Folder = "Default", image_size = 0,  Cam_Mat=0, Cam_Dist=0, rms_error=0, mean_error=0, FOV_Camera_Data = 0 ):
        Directory , Camera_Name = os.path.split(Parent_Folder)
        Save_Folder = f"{Parent_Folder}/Calibration_Data"
        CHECK_DIR = os.path.isdir(Save_Folder)

        if not CHECK_DIR:
            os.makedirs(Save_Folder)

        File_Name = f"{Camera_Name}_Parameters.yaml"
        Save_File = os.path.join(Save_Folder , File_Name)
        (Image_Width, Image_Height) = image_size
        Cam_Mat = Cam_Mat.tolist()
        Cam_Dist = Cam_Dist.tolist()
        fovx, fovy, focalLength, principalPoint, aspectRatio = FOV_Camera_Data
        Camera_Data = {
        'Camera' : Camera_Name,
        'Frame Width':Image_Width,
        'Frame Height':Image_Height,
        'fx' : Cam_Mat[0][0],
        'fy' : Cam_Mat[1][1],
        'cx' : Cam_Mat[0][2],
        'cy' : Cam_Mat[1][2],
        'k1' : Cam_Dist[0][0],
        'k2' : Cam_Dist[0][1],
        'k3' : Cam_Dist[0][2],
        'p1' : Cam_Dist[0][3],
        'p2' : Cam_Dist[0][4],
        'FOVX' : fovx,
        'FOVY' : fovy,
        'Focal Lenght' : focalLength,
        'principal Point X' : principalPoint[0],
        'principal Point Y' : principalPoint[1],
        'Aspect Ratio' : aspectRatio,
        'RMS_Error' : rms_error,
        'MEAN_Error' : mean_error,

        }
        yaml.sort_base_mapping_type_on_output = False

        with open(Save_File, 'w') as outfile:
            yaml.dump(Camera_Data, outfile, default_flow_style=False,sort_keys=False)

        print(f"camera parameters save done in file --->{Save_File}")
        return Save_File


    # saves calculated relative pose between the camera pair
    def Save_Stereo_Calib(self, RMS_error, Translation, angles, Save_File):
        Translation = np.squeeze(Translation).tolist()
        angles = angles.tolist()
        Stereo_Data = {
        'Tx' : Translation[0],
        'Ty' : Translation[1],
        'Tz' : Translation[2],
        'Rotx' : angles[0],
        'Roty' : angles[1],
        'Rotz' : angles[2],
        'RMS_Error' : RMS_error
        }
        yaml.sort_base_mapping_type_on_output = False

        with open(Save_File, 'w') as outfile:
            yaml.dump(Stereo_Data, outfile, default_flow_style=False,sort_keys=False)

        print(f"Stereo parameters save done in file --->{Save_File}")
        return Save_File
    

    # calculates the needed properties of the virtual production camera such as the horizontal and vertical fov, 
    def VP_Camera_Parameters(self, VP_Cam_Mat):
         
        data = cv2.calibrationMatrixValues(VP_Cam_Mat, self.FRAME_DIM, self.VP_Cam_sensor_width, self.VP_Cam_sensor_height)
        pixel_size = self.VP_Cam_sensor_width / self.VP_Cam_pixels_horizontal *1000 # um

        return data




