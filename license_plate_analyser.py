import cv2
import pandas as pd
import numpy as np


class LicensePlateAnalyser:
    def __init__(self, image, nightvision=False):
        self.image = image
        self.nightvision = nightvision
        self.image = cv2.resize( self.image, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
        self.h, self.w = self.image.shape[:2]
        self.image_area = self.h * self.w
        self.filtered_candidates = []
        self.canvas = np.zeros((self.h, self.w), np.uint8)
    
    def process_license_plate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.nightvision:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 2)
        else:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 26)
        return thresh
    
    def connected_component_analysis(self, binary_image):
        output = cv2.connectedComponentsWithStats(binary_image, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        for i in range(1, numLabels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 300: continue
            character_height = y + h
            if character_height < 50: continue
            bbox_area = w * h
            (cX, cY) = centroids[i]
            char_to_bbox_ratio = area / bbox_area
            if char_to_bbox_ratio < 0.25: continue
            char_to_plate_ratio = (area / self.image_area) * 100
            if char_to_plate_ratio > 20: continue
            char_ratio = w / h
            if not 0.2 < char_ratio < 2: continue
            componentMask = (labels == i).astype("uint8") * 255
            componentMask = componentMask[y:y+h, x: x+w]
            self.filtered_candidates.append({'x': x, 'y': y, 'w':w , 'h': h, 'cx': cX, 'cy': cY , 'mask': componentMask})
        
    
    def filter_candidates(self):
        candidates_df = pd.DataFrame(self.filtered_candidates).sort_values(by='x')
        self.median_height = candidates_df['h'].median()
        self.mean_height = candidates_df['h'].mean()
        self.height_threshold = self.median_height // 8
        
        # filter candidates
        temp = []
        for ii, row in candidates_df.iterrows():
            if abs(row['h'] - self.mean_height) < self.height_threshold:
                temp.append(dict(row))

        # update some values based on filtered candidates
        self.candidates_df = pd.DataFrame(temp).sort_values(by='x')
        
    def draw_canvas(self):
        
        for i, row in self.candidates_df.iterrows():
            char_mask = row['mask']
            char_width = row['w']
            char_height = row['h']
            char_y = row['y']
            char_x = row['x']
                
            # Place the character mask on the canvas at its original y-coordinate
            self.canvas[char_y:char_y + char_height, char_x:char_x + char_width] = char_mask

    def enlarge_to_boundaries(self):
        width_median = (self.candidates_df['h'].median() * 0.6)
        leftmost_x = self.candidates_df['x'].min()
        rightmost_x = self.candidates_df['x'].max() + self.candidates_df['w'].max()
        top_y = self.candidates_df['y'].min()
        bottom_y = self.candidates_df['y'].max() + self.candidates_df['h'].max()
        self.left_boundary_points = (top_y, bottom_y, leftmost_x, leftmost_x - width_median)
        self.left_boundary_roi = self.image[top_y: bottom_y, leftmost_x - width_median: leftmost_x]
        self.right_boundary_points = (top_y, bottom_y, rightmost_x, rightmost_x + width_median)
        self.right_boundary_roi = self.image[top_y: bottom_y, rightmost_x: rightmost_x + width_median]
        return self.left_boundary_points, self.left_boundary_roi, self.right_boundary_points, self.right_boundary_roi
