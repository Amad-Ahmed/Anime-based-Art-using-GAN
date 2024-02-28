# Anime-based-Art-using-GAN
 The problem addressed in this project revolves around generating Anime Faces using a GAN model with the additional capability of selecting hair color. Anime art style has gained significant popularity not only among enthusiasts but also in various industries, such as gaming and animated movie production. However, creating high-quality and diverse Anime Faces can be a time-consuming and challenging task for artists. 
The importance of this problem lies in the potential it holds for providing a valuable tool to artists and creators in the gaming and animated movie generation industry. By automating the process of generating Anime Faces, this project aims to offer a source of inspiration and assistance, ultimately enhancing the efficiency and creativity of artists. This tool could help reduce the time and effort required to create diverse character designs, enabling artists to focus more on other aspects of their work, such as storytelling and animation.
Furthermore, with the ability to select hair color, the project adds an additional layer of customization and flexibility for artists. This feature can facilitate the exploration of different character variations and help artists align their creations with specific themes, art styles, or narrative requirements.

# Dataset
The dataset used in this project is called "AnimeFacesDataset." It consists of a folder named "images," which contains over 65,000 Anime face images. The images in this dataset showcase a variety of characters with different hair colors.
In order to classify the Anime face images based on hair color, we utilized a preprocessing step involving k-means clustering. K-means clustering is an unsupervised learning algorithm that groups similar data points together. By applying k-means clustering to the dataset, we were able to identify the dominant colors in the hair regions of the images.
The result of the k-means clustering process provided us with the RGB (Red-Green-Blue) values representing the hair color of each image. However, in order to make the classification more interpretable and user-friendly, we integrated an open source model called the "Color Model." This model takes the RGB values as input and outputs the corresponding color class.
By incorporating the Color Model, we were able to convert the RGB values obtained from the k-means clustering step into more descriptive and meaningful color classes. This integration allowed for a more intuitive organization of the dataset, as each image was placed in a specific folder corresponding to its hair color class.
This preprocessing approach of combining k-means clustering with the Color Model not only provided a systematic way to classify the Anime face images based on hair color but also facilitated the subsequent user selection and customization of hair color during the generation process.

# Evaluation metrics
To evaluate the success of the generated Anime Faces and the overall performance of the model, we employ both qualitative and quantitative evaluation metrics. The metrics used include real and fake scores, as well as the losses of the generator and discriminator networks.

![evaluation metric 1](https://github.com/Amad-Ahmed/Anime-based-Art-using-GAN/assets/80278397/5095d054-8c7f-4d97-93e0-bb5bedb63c42)

![evaluation metric 2](https://github.com/Amad-Ahmed/Anime-based-Art-using-GAN/assets/80278397/e1b6917f-9c6b-40f8-9849-5792c1795b86)

# Output

![pic1](https://github.com/Amad-Ahmed/Anime-based-Art-using-GAN/assets/80278397/29180b85-cff4-464d-ae0c-99172fecec99)


![pic2](https://github.com/Amad-Ahmed/Anime-based-Art-using-GAN/assets/80278397/81b7cdcb-35a3-48d1-989b-3fc50ba3e3f4) 
