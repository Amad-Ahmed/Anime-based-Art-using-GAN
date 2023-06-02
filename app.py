from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QStackedWidget, QVBoxLayout, QMessageBox, QComboBox, QSlider, QFileDialog
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import QTimer, QSize, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sys
from os.path import exists
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import torchvision.utils as vutils
from PIL import Image, ImageFilter, ImageEnhance
import glob
import zipfile
import shutil


DICTIONARY = {
    "Random": "generators/generator100.pt",
    "Blue": "generators/generatorBlue.pt",
    "Green": "generators/generatorGreen.pt",
    "Pink": "generators/generatorPink.pt",
    "Purple": "generators/generatorPurple.pt",
    "Red": "generators/generatorRed.pt",
    "White": "generators/generatorWhite.pt",
    "Yellow": "generators/generatorYellow.pt",
    "Brown": "generators/generatorBrown.pt",
}

GENERATOR_PATH = None

VIDEO_PATH = []
VIDEO_CAPS = {}

class Window(QWidget):
    def __init__(self):
        super().__init__()

        def delete_folder(folder_path):
            try: 
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    os.remove(file_path)

                os.rmdir(folder_path)
            except: 
                pass

        delete_folder("generated_images")
        delete_folder("generated_videos")

        self.setWindowTitle("AnimeFacescape")
        # self.setWindowIcon(QIcon("icon.png"))
        self.setFixedHeight(700)
        self.setFixedWidth(1000)
        self.setContentsMargins(0,0,0,0)

        self.stackedWidget =  QStackedWidget()        
        self.firstPageWidget =  QWidget()
        self.secondPageWidget =  QWidget()
        self.stackedWidget.addWidget(self.firstPageWidget)
        self.stackedWidget.addWidget(self.secondPageWidget)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.stackedWidget)
        self.setLayout(layout)

        self.setStyleSheet("background-color: white")
        image = QImage('banner.jpeg')

        windowTitle = QLabel(self.firstPageWidget)
        windowTitle.setGeometry(270, 20, 1000, 100)
        windowTitle.setText("AnimeFacescape")
        windowTitle.setStyleSheet('''font-family: sans-serif;
            font-size: 50px;
            font-weight: bold;
            color: Blue; ''')


        label_logo = QLabel(self.firstPageWidget)
        label_logo.setGeometry(80, 140, 800, 168)
        label_logo.setPixmap(QPixmap(image).scaledToWidth(800))
        label_logo.show()

        x_val = -130

        select_label = QLabel(self.firstPageWidget)
        select_label.setGeometry(365 + x_val, 340, 250, 30)
        select_label.setText("Select number of images")
        select_label.setStyleSheet('''font-family: sans-serif;
            font-size: 20px;    
            color: black; ''')

        self.comboBox = QComboBox(self.firstPageWidget)
        self.comboBox.setGeometry(430 + x_val, 400, 100, 30)
        self.comboBox.addItem("2")
        self.comboBox.addItem("4")
        self.comboBox.addItem("6")

        x_val = 50

        select_label = QLabel(self.firstPageWidget)
        select_label.setGeometry(505 + x_val, 340, 250, 30)
        select_label.setText("Select hair color")
        select_label.setStyleSheet('''font-family: sans-serif;
            font-size: 20px;
            color: black; ''')
        

        self.comboBox_generator = QComboBox(self.firstPageWidget)
        self.comboBox_generator.setGeometry(530 + x_val, 400, 100, 30)
        self.comboBox_generator.addItem("Random")
        self.comboBox_generator.addItem("Yellow")
        self.comboBox_generator.addItem("Red")
        self.comboBox_generator.addItem("Blue")
        self.comboBox_generator.addItem("Green")
        self.comboBox_generator.addItem("Pink")
        self.comboBox_generator.addItem("Purple")
        self.comboBox_generator.addItem("White")
        self.comboBox_generator.addItem("Brown")
        

        btn = QPushButton("Generate", self.firstPageWidget)
        btn.setGeometry(410, 480, 150, 50)
        btn.clicked.connect(self.clicked_btn)
        btn.setStyleSheet('''width: 100px;
            height: 50px;
            border-radius: 15px;
            font-family: sans-serif;
            font-size: 20px;
            font-weight: bold;
            background-color: Blue; 
            color: white;''')
        btn.show()

        self.stackedWidget.setCurrentIndex(0)

        windowTitle2 = QLabel(self.secondPageWidget)
        windowTitle2.setGeometry(270, 10, 1000, 100)
        windowTitle2.setText("AnimeFacescape")
        windowTitle2.setStyleSheet('''font-family: sans-serif;
            font-size: 50px;
            font-weight: bold;
            color: Blue; ''')
        
    def clicked_btn(self): 
        global GENERATOR_PATH
        global VIDEO_PATH
        GENERATOR_PATH = DICTIONARY[self.comboBox_generator.currentText()]

        generate_images(int(self.comboBox.currentText()))
        generate_video()

        for i in range(int(self.comboBox.currentText())//2): VIDEO_PATH.append(f"generated_videos/{i*2}.avi")
        # print(VIDEO_PATH)

        self.stackedWidget.setCurrentIndex(1)

        no_of_columns = 3
        
        no_of_rows = int(self.comboBox.currentText()) // 2
        num = no_of_columns * no_of_rows
        self.labels = [QLabel(self.secondPageWidget) for i in range(num)]

        ymargin = 0        
        j = 0
        for i in range(num):
            
            if i % no_of_columns == 0: xmargin = 0
            self.labels[i].setGeometry(80 + (i%no_of_columns)*150 + xmargin, 150 + (i//no_of_columns)*150 + ymargin, 128, 128)
            self.labels[i].setStyleSheet("border: 1px solid black")
            
            if i != 2 and i != 5 and i != 8: 
                self.labels[i].setPixmap(QPixmap(f"generated_images/{j}.png").scaledToWidth(128))
                j += 1
            self.labels[i].show()

            xmargin += 25
            if i % no_of_columns == no_of_columns - 1: ymargin += 10

        downloadButtons = [QPushButton("Download", self.secondPageWidget) for i in range(no_of_rows)]
        for i in range(no_of_rows): 
            y_axis = 150 + i*150 + 50
            x_axis = 800

            downloadButtons[i].setGeometry(x_axis, y_axis, 150, 50)
            downloadButtons[i].setStyleSheet('''width: 100px;
            height: 50px;
            border-radius: 15px;
            font-family: sans-serif;
            font-size: 20px;
            font-weight: bold;
            background-color: Blue; 
            color: white;''')

            downloadButtons[i].show()
        
        if no_of_rows > 0: downloadButtons[0].clicked.connect(lambda: self.open_dialog(['generated_images/0.png', 'generated_images/1.png', 'generated_videos/0.avi']))
        if no_of_rows > 1: downloadButtons[1].clicked.connect(lambda: self.open_dialog(['generated_images/2.png', 'generated_images/3.png', 'generated_videos/2.avi']))
        if no_of_rows > 2: downloadButtons[2].clicked.connect(lambda: self.open_dialog(['generated_images/4.png', 'generated_images/5.png', 'generated_videos/4.avi']))

        self.video_widgets = [QVideoWidget(self.secondPageWidget) for i in range(no_of_rows)]
        self.media_players = [QMediaPlayer(self.secondPageWidget) for i in range(no_of_rows)]

        for i in range(no_of_rows): 
            y_axis = 150 + i*(150 + 10) 
            x_axis = 600

            # Create a video widget to display the video
            self.video_widgets[i].setGeometry(x_axis, y_axis, 128, 128)
            
            # Create a media player instance
            self.media_players[i].setVideoOutput(self.video_widgets[i])

            self.media_players[i].mediaStatusChanged.connect(self.handle_media_state_changed)

            # print(f"generated_videos/{i*2}.mp4")
            self.media_players[i].setMedia(QMediaContent(QUrl.fromLocalFile(f"generated_videos/{i*2}.avi")))

            self.video_widgets[i].show()
            self.media_players[i].play()

        self.sliders = [QSlider(self.secondPageWidget) for i in range(no_of_rows)]
        for i in range(no_of_rows):
            # print("iterating in sliders: ", i)
            y_axis = 150 + i*(150 + 10) + 128 + 10
            x_axis = 430

            self.sliders[i].setGeometry(x_axis, y_axis, 128, 10)
            self.sliders[i].setOrientation(1)
            self.sliders[i].setRange(0, 100)
            self.sliders[i].setValue(0)
            # self.sliders[i].valueChanged.connect(lambda index=i: self.slider_changed(index))

            self.sliders[i].show()

        if no_of_rows > 0: self.sliders[0].valueChanged.connect(lambda: self.slider_changed(0))
        if no_of_rows > 1: self.sliders[1].valueChanged.connect(lambda: self.slider_changed(1))
        if no_of_rows > 2: self.sliders[2].valueChanged.connect(lambda: self.slider_changed(2))

    def slider_changed(self, index):
        global VIDEO_PATH
        global VIDEO_CAPS

        # print("i = ", index)

        percentage = self.sliders[index].value()
        # print("video path : ", VIDEO_PATH[index])
        # print("percentage : ", percentage)

        if VIDEO_PATH[index] not in VIDEO_CAPS: VIDEO_CAPS[VIDEO_PATH[index]] = cv2.VideoCapture(VIDEO_PATH[index])

        self.get_video_frame(VIDEO_CAPS[VIDEO_PATH[index]], percentage)

        # no_of_rows = int(self.comboBox.currentText()) // 2

        if index == 0: j = 2
        elif index == 1: j = 5
        elif index == 2: j = 8

        self.labels[j].setPixmap(QPixmap("temp_frame.jpg").scaledToWidth(128))
        self.labels[j].setStyleSheet("border: 1px solid red") 

    def get_video_frame(self, cap, percentage):
        # cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int((percentage / 100) * total_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            # raise ValueError("Unable to read video frame.")
            return
        
        # cap.release()
        cv2.imwrite("temp_frame.jpg", frame)
        # return frame

    def handle_media_state_changed(self, state):
        if state == QMediaPlayer.EndOfMedia:
            for media_player in self.media_players:
                media_player.setPosition(0)
                media_player.play()

    def open_dialog(self, files):

        create_zip(files, "animefacescape.zip")

        file_dialog = QFileDialog()
        folder_path = file_dialog.getExistingDirectory(None, "Select Folder", options=QFileDialog.ShowDirsOnly)
        if folder_path:
            source_file_path = 'animefacescape.zip'
            file_name = os.path.basename(source_file_path)  # Extract the filename from the source file path
            destination_path = os.path.join(folder_path, file_name)  # Build the destination file path
            
            try:
                shutil.move(source_file_path, destination_path)  # Move the file to the destination folder
                print("File moved successfully.")
            except Exception as e:
                # print("Error occurred while moving the file:", str(e))
                QMessageBox.about(self, "Error", "Error occurred while moving the file: " + str(e))
        else: 
            pass


def generate_images(no_of_images):
    global GENERATOR_PATH
    # Define the generator model
    latent_size = 128

    generator = nn.Sequential(
        nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
        nn.Tanh()
    )

    # Load the generator model from the saved file
    generator_state_dict = torch.load(GENERATOR_PATH)
    new_generator = nn.Sequential()
    for i, layer in enumerate(generator):
        if isinstance(layer, nn.ConvTranspose2d):
            # get the conv layer weights from the state dict and add to the new generator
            conv_layer = nn.ConvTranspose2d(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, bias=False)
            conv_layer.weight.data = generator_state_dict[f'{i}.weight']
            new_generator.add_module(f'conv{i}', conv_layer)
            
        elif isinstance(layer, nn.BatchNorm2d):
            # get the batch norm layer weights from the state dict and add to the new generator
            bn_layer = nn.BatchNorm2d(layer.num_features)
            bn_layer.weight.data = generator_state_dict[f'{i}.weight']
            bn_layer.bias.data = generator_state_dict[f'{i}.bias']
            bn_layer.running_mean.data = generator_state_dict[f'{i}.running_mean']
            bn_layer.running_var.data = generator_state_dict[f'{i}.running_var']
            new_generator.add_module(f'bn{i}', bn_layer)
            
        elif isinstance(layer, nn.ReLU):
            new_generator.add_module(f'relu{i}', nn.ReLU(True))
            
        elif isinstance(layer, nn.Tanh):
            new_generator.add_module(f'tanh{i}', nn.Tanh())

    generator = new_generator

    # Create a directory to save the generated images
    os.makedirs('generated_images', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate 10 random latent vectors
    latent_vectors = torch.randn(no_of_images, latent_size, 1, 1)
    latent_vectors = latent_vectors.to(device)

    # Generate the images from the latent vectors using the generator
    with torch.no_grad():
        generated_images = generator(latent_vectors)
        # print(generated_images.shape)

    # Save each generated image separately in the folder
    for i in range(generated_images.shape[0]):
        image = generated_images[i].permute(1, 2, 0).detach().cpu().numpy()
        image = (image + 1) / 2.0 * 255.0
        image = image.astype('uint8')
        image = Image.fromarray(image)

        # Upscale the image to 512x512 using bicubic interpolation
        upscaled_image = image.resize((128, 128), resample=Image.BICUBIC)

        # # Increase the contrast by a factor of 1.5
        # contrasted_image = ImageEnhance.Contrast(upscaled_image).enhance(1.5)

        # # Increase the sharpness by a factor of 1.5
        # sharpened_image = ImageEnhance.Sharpness(contrasted_image).enhance(1.5)

        # sharpened_array = np.asarray(sharpened_image)

        # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

        # sharpened_array = cv2.filter2D(sharpened_array, -1, sharpen_kernel)
        # sharpened_image = Image.fromarray(sharpened_array)

        # sharpened_image.save(f'generated_images/{i}.png')
        upscaled_image.save(f'generated_images/{i}.png')

    return 1

def generate_video():
    # Set the path to the folder where the generated images are located
    img_folder = "generated_images/"

    # Get the list of all the image file names in the folder
    img_names = sorted(glob.glob(os.path.join(img_folder, "*.png")))

    # Set the size of the output video and the frame rate
    frame_size = (128, 128)
    fps = 10.0

    # Set the transition duration in seconds
    transition_time = 5.0

    # Calculate the number of frames for the transition
    transition_frames = int(transition_time * fps)

    # make dir
    os.makedirs('generated_videos', exist_ok=True)

    # Loop through the list of images in steps of 2 and create a new video file for every pair of consecutive images
    for i in range(0, len(img_names) - 1, 2):
        # Create an empty list to store all the images
        images = []

        # Read the current and next images using cv2.imread() and append them to the list
        curr_img = cv2.imread(img_names[i])
        next_img = cv2.imread(img_names[i + 1])
        curr_img2 = cv2.imread(img_names[i])
        images.append(curr_img)
        images.append(next_img)
        images.append(curr_img2)

        # Create the video writer object
        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        
        # store inside generated_videos
        # video_name = f"generated_videos/{i}.mp4"
        video_name = f"generated_videos/{i}.avi"

        video = cv2.VideoWriter(video_name, fourcc, fps, frame_size, isColor=True)

        # Set the codec and bitrate for the video writer object
        # video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"avc1"))
        # video.set(cv2.CAP_PROP_BITRATE, 10000)
        video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        video.set(cv2.CAP_PROP_BITRATE, 10000)


        # # Loop over the transition frames and blend the images using linear cross-fade
        # for j in range(transition_frames):
        #     # Calculate the alpha value for the current frame
        #     alpha = float(j) / float(transition_frames)

        #     # Linearly interpolate between the two images
        #     blended_img = cv2.convertScaleAbs((1.0 - alpha) * curr_img + alpha * next_img)

        #     # Write the blended image to the video file
        #     video.write(cv2.resize(blended_img, frame_size))

        # # Write the next image to the video file without any blending
        # video.write(cv2.resize(next_img, frame_size))

        # for j in range(transition_frames):
        #     # Calculate the alpha value for the current frame
        #     alpha = float(j) / float(transition_frames)

        #     # Linearly interpolate between the two images
        #     blended_img = cv2.convertScaleAbs((1.0 - alpha) * next_img + alpha * curr_img2)

        #     # Write the blended image to the video file
        #     video.write(cv2.resize(blended_img, frame_size))
        
        # video.write(cv2.resize(curr_img2, frame_size))

                # Loop over the transition frames and blend the images using linear cross-fade
        for j in range(transition_frames):
            # Calculate the alpha value for the current frame
            alpha = float(j) / float(transition_frames)

            # Linearly interpolate between the two images
            blended_img = cv2.convertScaleAbs((1.0 - alpha) * curr_img + alpha * next_img)

            # Write the blended image to the video file
            video.write(cv2.resize(blended_img, frame_size))

        # Write the next image to the video file without any blending
        video.write(cv2.resize(next_img, frame_size))

        for j in range(transition_frames):
            # Calculate the alpha value for the current frame
            alpha = float(j) / float(transition_frames)

            # Linearly interpolate between the two images
            blended_img = cv2.convertScaleAbs((1.0 - alpha) * next_img + alpha * curr_img2)

            # Write the blended image to the video file
            video.write(cv2.resize(blended_img, frame_size))

        video.write(cv2.resize(curr_img2, frame_size))

        # Release the video writer object
        video.release()
    
    return 1


def create_zip(file_paths, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for file_path in file_paths:
            zip_file.write(file_path)

app = QApplication([])
window = Window()
window.setWindowIcon(QIcon("icon.jpg"))
window.show()
sys.exit(app.exec())

