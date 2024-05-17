#!/usr/bin/env python3

#ROS imports
# import rospy
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import TwistStamped
# from std_msgs.msg import Float64

import numpy as np
from numpy.random import uniform, randn
import matplotlib.pyplot as plt
import cv2
from scipy.stats import norm
from filterpy.monte_carlo import multinomial_resample
# import numexpr as ne
from numba import jit
import time

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


#cv bridge to convert ros img to opencv img
# bridge = CvBridge()
velocity = [1,1]
altitude = 100

# params

fov = 100
angle = fov/2
width_multiplier = 2 * np.tan(np.radians(angle))
height_multiplier = (3 * width_multiplier)/4

plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y)
plt.xlim(-150,150)
plt.ylim(-150,150)
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

@jit(nopython=True)
def get_encode(pts, W):
    Z = np.exp(1j * pts @ W).sum(axis=0)
    Z = Z / np.linalg.norm(Z)
    return Z

class Encoder:
    def __init__(self):
        self.d = 128
        self.alpha = 0.1
        self.W = np.stack([self.strict_standard_normal(self.d) for _ in range(2)], axis=0) * self.alpha
        self.W = self.W.astype(np.complex128)
        self.detector = cv2.SIFT_create(20)
        
    def strict_standard_normal(self, d):
        y = np.linspace(0, 1, d+2)
        x = norm.ppf(y)[1:-1]
        np.random.shuffle(x)
        return x
    
    def encode(self, img):
        kp = self.detector.detect(img, None)
        pts = cv2.KeyPoint_convert(kp).astype(np.complex128)
        return get_encode(pts,self.W)

    def similarity(self, x, y):
        return np.absolute(np.sum(x * y.conj()))

class DLCosineEncoder:
    def __init__(self):
        # Load the pretrained model
        self.model = models.resnet18(pretrained=True)
        # Use the model object to select the desired layer
        self.layer = self.model._modules.get('avgpool')

        # Set model to evaluation mode
        self.model.eval()
        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()
        self.sift_function = cv2.SIFT_create(nfeatures=500)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    def get_kpt_img(self, rgb_img):
        gray_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY) # Converting the RGB image to grayscale
         # Creating an instance of the SIFT Function
        keypoints, features = self.sift_function.detectAndCompute(gray_img,None) # Computing the set of keypoints and features for the image
        blank = np.zeros(rgb_img.shape).astype(np.uint8)
        for i in range(len(keypoints)):
    #         print(int(keypoints[i].pt[1]),int(keypoints[i].pt[0]))
            cv2.circle(blank, (int(keypoints[i].pt[0]),int(keypoints[i].pt[1])), 3, (255,255,255), -1)
    #         blank[int(keypoints[i].pt[1]),int(keypoints[i].pt[0])] = [255,255,255]
        return blank
    
    def encode(self, img):
        # 2. Create a PyTorch Variable with the transformed image
        t_img = Variable(self.normalize(self.scaler(self.to_tensor(img))).unsqueeze(0))
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(1,512,1,1)
        # 4. Define a function that will copy the output of a layer

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)
        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        self.model(t_img)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector

        return my_embedding.view(512)
    
    def similarity(self, x, y):
        return self.cos(x.unsqueeze(0),y.unsqueeze(0))


class ParticleFilter:
    def __init__(self, image_path, num_particles, initial_state, process_noise_std):
        self.num_particles = num_particles
        #known init position
        self.particles = self.create_gaussian_particles(initial_state, process_noise_std, self.num_particles)
        # self.particles = self.create_uniform_particles((-100,100), (-100,100), self.num_particles)
        #plot
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        sc.set_offsets(np.c_[x,y])
        fig.canvas.draw_idle()
        plt.pause(0.01)

        self.weights = np.ones(num_particles) / num_particles
        # self.encoder = Encoder()
        self.encoder = DLCosineEncoder()
        # self.map_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        self.map_img = cv2.imread(image_path)
        self.img_sz = self.map_img.shape[0]
        self.map_sz = 1200
        self.resolution = self.img_sz/self.map_sz

    def create_uniform_particles(self, x_range, y_range, N):
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

    def measurement_model(self, encoding1, img2):
        # Compute image similarity using feature encodings
        encoding2 = self.encoder.encode(img2)
        return self.encoder.similarity(encoding1, encoding2)

    def resample_from_index(self, particles, weights, indexes):
        self.particles[:] = particles[indexes]
        self.weights.resize(len(particles))
        self.weights.fill(1.0 / len(weights))

    def update(self, img):
        # Update particle weights based on measurement
        encoding1 = self.encoder.encode(img)
        for i in range(self.num_particles):
            # Generate a predicted image based on the current particle's state
            predicted_img = self.generate_predicted_image(self.particles[i], self.map_img)
            
            if predicted_img is not None:
                #similarity score is the difference in measurement (measurement - predicted measurement)
                diff_measurement = self.measurement_model(encoding1, predicted_img)
                # self.weights[i] *= np.exp(-0.5 * (diff_measurement)**2) # Update the particle's weight based on the difference between predicted and actual measurements
                self.weights[i] = diff_measurement
            else:
                return
        # self.weights += 1.e-300 
        # self.weights = (self.weights - min(self.weights))/ (max(self.weights)-min(self.weights))
        self.weights += 1.e-300 
        self.weights /= np.sum(self.weights) # Normalize the weights
        indexes = multinomial_resample(self.weights)
        self.resample_from_index(self.particles, self.weights, indexes)

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
        pos = self.particles[:, 0:2]
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
        current_image = cv2.cvtColor(cv2.resize(drone_view, 
                                                (img_width, img_height), 
                                                interpolation = cv2.INTER_LINEAR)
                                    , cv2.COLOR_BGR2GRAY)
        # cv2.imwrite("test2.jpg", current_image)

def altitude_callback(msg):
    global altitude
    altitude = msg.data

def main():
    global velocity, current_image
    #ros stuff
    # rospy.init_node('particle_filter', anonymous=True)
    # rospy.Subscriber("/drone_camera/image_raw", Image, image_callback)
    # rospy.Subscriber("/mavros/global_position/raw/gps_vel", TwistStamped, velocity_callback)
    # rospy.Subscriber("/mavros/global_position/rel_alt", Float64, altitude_callback)

    image_path = "map_sq.png"
    num_particles = 400
    initial_state = np.array([0, 0])  # Initial guess of drone position (pixel coordinates)
    process_noise_std = np.array([20, 20]) 
    particle_filter = ParticleFilter(image_path, num_particles, initial_state, process_noise_std)
    time.sleep(0.2)
    # Main loop to update and predict
    for i in range(200):  # Simulate 20 frames
        x = particle_filter.particles[:, 0]
        y = particle_filter.particles[:, 1]
        sc.set_offsets(np.c_[x,y])
        fig.canvas.draw_idle()
        plt.pause(0.01)
        estimated_position, estimated_variance = particle_filter.get_estimate()
        print("Estimated Drone Position:", estimated_position)
        particle_filter.predict(velocity)
        print("Ground truth", initial_state)
        current_image = particle_filter.generate_predicted_image(initial_state, particle_filter.map_img)
        initial_state[:]+= 1
        particle_filter.update(current_image)
    
    plt.waitforbuttonpress()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
