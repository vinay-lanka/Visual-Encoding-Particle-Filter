#!/usr/bin/env python3

#ROS imports
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64

import os
import numpy as np
from numpy.random import uniform, randn
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import copy

#cv bridge to convert ros img to opencv img
bridge = CvBridge()
velocity = [0,0]
altitude = 0

# params

fov = 100
angle = fov/2
width_multiplier = 2 * np.tan(np.radians(angle))
height_multiplier = (3 * width_multiplier)/4

plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y)
plt.xlim(-100,100)
plt.ylim(-100,100)

plt.draw()

class ImageEncoder():

    def __init__(self,image_path,siftD=128,num_clusters=64):
        self.siftD = siftD
        self.num_clusters = num_clusters
        self.clustering_model = self.create_hist_model(image_path)

    def detect_sift_features(self,rgb_img):
        gray_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
        sift_function = cv2.SIFT_create(nfeatures=128)
        keypoints, features = sift_function.detectAndCompute(gray_img,None)
        return keypoints, features

    def create_hist_model(self,image_path):
        all_features = []
        img = cv2.imread(image_path)
        keypoints, features = self.detect_sift_features(img)
        all_features.append(features)
        features_array = np.concatenate(all_features,axis=0).reshape((-1,self.siftD))
        clustering_model = KMeans(n_clusters=64,n_init="auto")
        clustering_model.fit(features_array)
        return clustering_model

    def hist_encode(self,rgb_img):
        keypoints, features = self.detect_sift_features(rgb_img)
        features = np.array(features).reshape((-1,self.siftD))
        feature_labels = self.clustering_model.predict(features).reshape((-1))
        hist, bin_edges = np.histogram(feature_labels,bins=[i for i in range(self.num_clusters)])
        return np.array(hist).astype(np.float32)

    def hist_similarity_correlation(self,hist1,hist2):
        sim = cv2.compareHist(hist1,hist2,0)
        return sim

    def hist_similarity_intersection(self,hist1,hist2):
        sim = cv2.compareHist(hist1,hist2,2)
        return sim

class ParticleFilter:
    def __init__(self, image_path, num_particles, initial_state, process_noise_std):
        self.num_particles = num_particles
        #old gaussian sampling
        # self.particles = np.array([initial_state + np.random.randn(2) * process_noise_std for _ in range(num_particles)])
        #old uniform sampling
        # x = np.random.randint(-200, 200, self.num_particles)
        # y = np.random.randint(-200, 200, self.num_particles)
        # self.particles = np.vstack((x, y)).T

        self.particles = self.create_gaussian_particles(initial_state, process_noise_std, self.num_particles)

        x = self.particles[:, 0]
        y = self.particles[:, 1]
        sc.set_offsets(np.c_[x,y])
        fig.canvas.draw_idle()
        plt.pause(0.1)

        self.weights = np.ones(num_particles) / num_particles
        self.encoder = ImageEncoder(image_path)
        self.map_img = cv2.imread(image_path)
        self.img_sz = self.map_img.shape[0]
        self.map_sz = 1200
        self.resolution = self.img_sz/self.map_sz

    def create_uniform_particles(self, x_range, y_range, hdg_range, N):
        particles = np.empty((N, 2))
        particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
        particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
        return particles

    def create_gaussian_particles(self, mean, std, N):
        particles = np.empty((N, 2))
        particles[:, 0] = mean[0] + (randn(N) * std[0])
        particles[:, 1] = mean[1] + (randn(N) * std[1])
        return particles

    def predict(self, velocity):
        # Update particle positions based on constant velocity motion model (Change this motion model)
        velocity_arr = np.full((len(self.particles), 2), velocity)
        self.particles =  self.particles + velocity_arr

    def measurement_model(self, img1, img2):
        # Compute image similarity using feature encodings
        encoding1 = self.encoder.hist_encode(img1)
        encoding2 = self.encoder.hist_encode(img2)
        return self.encoder.hist_similarity_correlation(encoding1, encoding2)

    def resample_particles(self):
        # Resample particles based on weights
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = np.array([self.particles[i] for i in indices])
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def update(self, img):
        # Update particle weights based on measurement
        for i in range(self.num_particles):
            # Generate a predicted image based on the current particle's state
            predicted_img = self.generate_predicted_image(self.particles[i], self.map_img)
            if predicted_img is not None:
                #similarity score is the difference in measurement (measurement - predicted measurement)
                diff_measurement = self.measurement_model(predicted_img, img)
                # self.weights[i] *= np.exp(-0.5 * (diff_measurement)**2) # Update the particle's weight based on the difference between predicted and actual measurements
                self.weights[i] = diff_measurement
            else:
                return
        print(self.weights)
        self.weights /= np.sum(self.weights) # Normalize the weights
        self.resample_particles()

    def state_to_pixel(self, state):
        predicted_x, predicted_y = state
        predicted_x_px = int((self.img_sz/2) + (self.resolution * predicted_x))
        predicted_y_px = int((self.img_sz/2) + (self.resolution * predicted_y))
        return predicted_x_px, predicted_y_px

    def generate_predicted_image(self, state, img):
        # Generate predicted image based on particle state
        predicted_x_px, predicted_y_px = self.state_to_pixel(state)
        img_width = (altitude * width_multiplier)
        img_height = (altitude * height_multiplier)
        if img_width>0:
            # return img[predicted_y_px-int(img_height/2):predicted_y_px+int(img_height/2), 
            #         predicted_x_px-int(img_width/2):predicted_x_px+int(img_width/2)] 
            return cv2.rotate(img[predicted_x_px-int(img_width/2):predicted_x_px+int(img_width/2),
                                predicted_y_px-int(img_height/2):predicted_y_px+int(img_height/2)], 
                                cv2.ROTATE_90_COUNTERCLOCKWISE)   
        else:
            return None
        
    def get_estimate(self):
        # Estimate drone position by averaging particles weighted by their importance
        # return np.average(self.particles, axis=0, weights=self.weights)
        pos = self.particles[:, 0:1]
        mean = np.average(pos, weights=self.weights, axis=0)
        var  = np.average((pos - mean)**2, weights=self.weights, axis=0)
        return mean, var

def velocity_callback(msg):
    global velocity
    velocity[0] = msg.twist.linear.x
    velocity[1] = msg.twist.linear.y

def image_callback(msg):
    global current_image, altitude
    drone_view = bridge.imgmsg_to_cv2(msg, "bgr8")
    img_width = int(altitude * width_multiplier)
    img_height = int(altitude * height_multiplier)
    if img_width>0:
        current_image = cv2.resize(drone_view, (img_width, img_height), interpolation = cv2.INTER_LINEAR)
        # cv2.imwrite("test2.jpg", current_image)

def altitude_callback(msg):
    global altitude
    altitude = msg.data

def main():
    global velocity, current_image
    #ros stuff
    rospy.init_node('particle_filter', anonymous=True)
    rospy.Subscriber("/drone_camera/image_raw", Image, image_callback)
    rospy.Subscriber("/mavros/global_position/raw/gps_vel", TwistStamped, velocity_callback)
    rospy.Subscriber("/mavros/global_position/rel_alt", Float64, altitude_callback)

    image_path = "map_sq.png"
    num_particles = 200
    initial_state = np.array([0, 0])  # Initial guess of drone position (pixel coordinates)
    process_noise_std = np.array([10, 10]) 
    particle_filter = ParticleFilter(image_path, num_particles, initial_state, process_noise_std)

    # # Main loop to update and predict
    # for i in range(2):  # Simulate 20 frames
    #     x = particle_filter.particles[:, 0]
    #     y = particle_filter.particles[:, 1]
    #     sc.set_offsets(np.c_[x,y])
    #     fig.canvas.draw_idle()
    #     plt.pause(0.1)
    #     estimated_position, estimated_variance = particle_filter.get_estimate()
    #     print("Estimated Drone Position:", estimated_position)
    #     particle_filter.predict(velocity)
    #     particle_filter.update(current_image)
    
    plt.waitforbuttonpress()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
