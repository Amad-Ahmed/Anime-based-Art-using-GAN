import os
import mat
import color_classifier

folder_path = 'animefacedataset/images/'  # Replace with the path to your folder

for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_file = os.path.join(folder_path, filename)
        # Process each image file here
        # print(image_file)

        color = mat.extract_dominant_color(image_file, k=1)

        color_name = color_classifier.predict_color(color[0], color[1], color[2])

        path = 'animefacedataset/' + color_name

        # make directory if not exists
        if not os.path.exists(path):
            os.makedirs(path)

        # move file to directory
        os.rename(image_file, path + '/' + filename)





