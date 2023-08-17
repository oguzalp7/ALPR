import cv2
import numpy as np
import pandas as pd

from vehicle_detector import VehicleDetector
from vehicle_pose_detector import VehiclePoseDetector
from license_plate_detector import LicencePlateDetector
from license_plate_analyser import LicensePlateAnalyser
from ocr import OCR

class ALPR:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.vd = VehicleDetector()
        self.vpd = VehiclePoseDetector()
        self.lpd = LicencePlateDetector()
        self.ocr = OCR()
        self.nightvision = None

    def extract_text(self, image):
        vehicles = self.vd.detect_vehicles(image)
        if len(vehicles) > 0:
            self.nightvision = False
            if self.verbose:
                print('Multiple Vehicles detected.')
                print('Filtering vehicles, extracting the closest vehicle')
            # get the biggest bbox, AKA closest vehicle
            interested_vehicle = self.vd.generate_vehicle_roi(image, vehicles, debug=False)
            if self.verbose:
                print('Closest vehicle found.')
                print('Estimating the pose of the vehicle')
            pose_bbox, pose = self.vpd.get_best_results(interested_vehicle, debug=False)
            if self.verbose:
                print(f'Detected pose: {pose}')
            if pose == 'front':
                if self.verbose:
                    print('Detecting the License Plate...')
                licence_plates = self.lpd.detect_licence_plate(interested_vehicle)
                if licence_plates != 0:
                    if self.verbose:
                        print('License plate(s) detected. ')
                    licence_plate = self.lpd.generate_licence_plate_roi(interested_vehicle, licence_plates, debug=False)
                else:
                    if self.verbose:
                        print('License plate is not visible.')
                        print('Please warn the driver, so camera can detect license plate.')
                    license_plate = None
            else:
                print('The vehicle is exiting.')
                licence_plate = None
                return False, ""
        else:
            if self.verbose:
                print('Any vehicle detected.')
                print('Switching to nightmode.')
            h, w, _ = image.shape
            image = image[:, w//3: w]
            self.nightvision = True
            if self.verbose:
                print('Nightvision mode turned on.')
                print('Trying to find license plates.')
            licence_plates = self.lpd.detect_licence_plate(image)
            if self.verbose:
                print(f'The number of license plates found: {len(licence_plates)}')
            if len(licence_plates) >= 1:
                print('License plate(s) detected.')
                print('Extracting the closest license plate.')
                licence_plate = self.lpd.generate_licence_plate_roi(image, licence_plates, debug=False)
            else:
                print('No license plate detected.')
                licence_plate = None
                return False, ""
            
        if licence_plate is not None:
            if self.verbose:
                print('License plate detection complete.')
                print('Processing license plate...')
            lpa = LicensePlateAnalyser(licence_plate, nightvision=self.nightvision)
            processed_license_plate = lpa.process_license_plate(image=lpa.image)
            # apply connected component anaysis => updates lpa.filtered_candidates
            lpa.connected_component_analysis(processed_license_plate)
            # apply filtering => generates a dataframe => lpa.candidates_df
            lpa.filter_candidates()
            lpa.draw_canvas()
            return True, self.ocr.read_text(lpa.canvas)
        else:
            return False, ""