import streamlit as st
import json
from streamlit_lottie import st_lottie

def projects():
    col1, col2 = st.columns(2)

    # col1.markdown("## Experience")
    with col1:
        st.markdown("""
                <style>
                .centered {
                    display: flex;
                    align-items: center;
                    height: 100%;
                    margin-top: 200px; /* Adjust this value as needed */
                }
                </style>
                <div class="centered">
                    <h2>Projects </h2>
                </div>
            """, unsafe_allow_html=True)
    path = "Animation_girl.json"
    with open(path, "r") as file:
        url = json.load(file)
    with col2:
        st_lottie(url,
                  reverse=True,
                  height=400,
                  width=400,
                  speed=1,
                  loop=True,
                  quality='high',
                  )

    with st.container():
        col1,col2 = st.columns(2)
        with col1:
            with st.container(border=True):


                # Displaying the title of the project
                st.title("EMOTION-MARK-AI (FACIAL SENTIMENT ANALYSIZED ATTENDANCE TRACKER) ")

                # Displaying the description
                st.markdown("""
                **Description:**
                EMOTION-MARK-AI is an innovative attendance tracking system that leverages facial recognition and sentiment analysis to streamline and enhance the process of monitoring attendance in educational institutions, workplaces, and events. By combining artificial intelligence and computer vision, this project offers a seamless, contactless, and efficient alternative to traditional attendance systems.
                
                - **Facial Recognition:** Identifies individuals based on facial features using machine learning, ensuring accurate and secure attendance tracking.

                - **Sentiment Analysis:** Analyzes facial expressions to gauge participants' emotional states, offering insights into engagement and mood.

                - **Real-time Tracking:** Processes video input instantly, marking attendance without manual intervention.

                - **Data Visualization:** Displays attendance and sentiment trends via interactive dashboards for clear insights.

                - **Secure and Scalable:** Safely stores facial data with privacy considerations and scales to support large groups.


                """)

                # Displaying the tools used
                st.markdown("""
                **Tools Used:**
                
                **Python** ,
                **libraries such as OpenCV, dlib, and face_recognition**,
                **Streamlit** ,
                **Machine learning**, 
                **CNN-based facial recognition model**, 
                **Sqlite3 for database**, 
                
                """)
                st.markdown(""" """)


                c1,c2,c3,c4 = st.columns(4)
                c1.markdown("""**[Link to app](https://ravinder.streamlit.app/)**  """)
                c2.markdown("""**[GitHub](https://github.com/ravinder2023/facial-sentiment-analysed-ai-attendance-tracker/blob/main)**""")
                #c3.markdown("""**[LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7220172770226102272/)** """)
                #c4.markdown("""**[X](https://x.com/streamlit/status/1814406829075542029)**""")
                #rc1,rc2 = st.columns(2)
                #rc1.markdown("""**[Streamlit community](https://buff.ly/3WqhYiB)**""")
                #rc2.markdown("""**[YouTube](https://www.youtube.com/watch?v=dwlE4p2uF6k)**""")


        with col2:
            with st.container(border=True):
                st.markdown(""" """)
                
                # Displaying the title of the project
                st.title("Autism Spectrum Disorder Prediction")
                st.markdown(""" """)
                st.markdown(""" """)


                # Displaying the description
                st.markdown("""
                **Description:**
                Autism Spectrum Disorder (ASD) Prediction is an AI-powered solution designed to detect early signs of ASD by analyzing behavioral, genetic, and environmental data. The project aims to facilitate timely diagnosis and intervention, improving the quality of life for individuals with ASD and their families.
               
                - **Behavioral Analysis:** Identifies ASD patterns in social interactions, communication, and repetitive behaviors using machine learning.

                - **Data Integration:** Combines facial expression, eye tracking, and speech data for comprehensive assessment.

                - **Early Detection:** Offers predictive insights to enable timely therapies and support systems.

                - **User-friendly Interface:** Provides interactive dashboards with risk scores and detailed reports.

                - **Secure and Private:** Ensures sensitive health data is stored securely and handled with compliance.                """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)


                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python**, 
                **Streamlit**,  **machine learning**, **Pandas** ,**Scikit-learn** ,**numpy** ,**tensorflow**)
                
                """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)


                c1, c2 = st.columns(2)
                c1.markdown("""**[Link to app](https://autism-prediction-raavi.streamlit.app/Form)**  """)
                c2.markdown("""**[GitHub](https://github.com/ravinder2023/Autism-prediction/blob/main)**""")
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                
                
                


        with col1:
            with st.container(border=True):
                # Displaying the title of the project
                st.title("Milk Quality Prediction Using Machine Learning")

                # Displaying the description
                st.markdown("""
                **Description:**
                 This project aims to automate the prediction of milk quality using machine learning. The system evaluates milk quality based on multiple factors, such as pH level, temperature, fat content, odor, and other relevant physical and chemical attributes. The primary goal is to classify the milk into different quality grades: High, Average, or Low.
                - **Data Input:** The user provides data through an interactive web form, including parameters like pH, temperature, taste, odor, fat content, turbidity, and color.
                - **Machine Learning Model:** A Gradient Boosting Classifier is used to train the model, based on a dataset that includes historical milk samples and their corresponding quality grades.
                - **Prediction Process:** Once the user inputs the data, the model makes a prediction based on the provided values and outputs the quality grade of the milk.
                - **Result Display:** The result is displayed dynamically, indicating whether the milk is of High, Average, or Low quality, allowing users to instantly assess the milk quality.
                """)
                st.markdown(""" """)



                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python**, **Pandas**, **Flask**, **Html**, **Css**,  **NumPy**, **Matplotlib**,  **Seaborn**, **Scikit-learn**
                """)
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)
                

                # Adding the GitHub link
                c1, c2 = st.columns(2)
                c1.markdown("""**[Link to app](https://milkquality.glitch.me/)**  """)
                c2.markdown("""**[Glitch code](https://glitch.com/edit/#!/milkquality?path=app.py%3A72%3A4)**""")
                st.markdown(""" """)
                st.markdown(""" """)
                st.markdown(""" """)


#         with col2:
#             with st.container(border=True):
#                 # Displaying the title of the project
#                 st.title("Generative AI text to image converter")

#                 # Displaying the description
#                 st.markdown("""
#                 **Description:**
#                 The Text-to-Image Converter is a machine learning-based application that generates images from textual descriptions. By utilizing Generative AI, this tool transforms natural language input into realistic images, allowing users to create visuals from written prompts.
                
#                 This project involves four parts:
# User Input: The user provides a textual description of the image they want to generate. For example, you can input something like, "A serene sunset over a beach with calm waves."
# AI Model: The system processes the textual input using a generative AI model trained on large datasets of images and their corresponding textual descriptions.
# Image Generation: The model creates a visually accurate image based on the description, capturing key visual elements and styles from the text prompt.
# Interactive Interface: The application is user-friendly, with a simple web interface that allows users to easily input their text and view the generated image in real-time.  """)
#                 st.markdown(""" """)


#                 # Displaying the tools used
#                 st.markdown("""
#                 **Tools Used:**

#                 **Python** , **NumPy**, **TensorFlow**, **Keras**,  **Matplotlib**
#                 """)
#                 st.markdown(""" """)


#                 # Adding the GitHub link
#                 st.markdown("""**[GitHub](https://github.com/archanags001/coding_challenge/blob/main/Coding_challenge_.ipynb)**""")

        # with col2:
        #     with st.container(border=True):
        #         st.markdown(""" """)

        #         # Displaying the title of the project
        #         st.title("Multiple LSTMs")
        #         st.markdown(""" """)
                



        #         # Displaying the description
        #         st.markdown("""
        #         **Description:**
        #         The "Multiple LSTMs" project focuses on building and comparing multiple Long Short-Term Memory (LSTM) models for time series forecasting. The project involves:

        #         - **Data Preparation:** Loading and preprocessing time series data.
        #         - **LSTM Model Implementation:** Creating and training multiple LSTM models.
        #         - **Model Evaluation:** Comparing the performance of the LSTM models.
        #         - **Visualization:** Plotting results to visualize model performance.
        #         """)
        #         st.markdown(""" """)
        #         st.markdown(""" """)
                


        #         # Displaying the tools used
        #         st.markdown("""
        #         **Tools Used:**

        #         **Python**,
        #         **TensorFlow**, **Keras**, 
        #         **NumPy**, 
        #         **Matplotlib**,
        #         """)
                
        #         st.markdown(""" """)
        #         st.markdown(""" """)




        #         # Adding the GitHub link
        #         st.markdown("""**[GitHub](https://github.com/archanags001/coding_challenge/blob/main/Multiple_LSTMs.ipynb)**""")
        #         st.markdown(""" """)
        #         st.markdown(""" """)
        #         st.markdown(""" """)



        # with col1:
        #     with st.container(border=True):

        #         # Displaying the title of the project
        #         st.title("TensorFlow Projects")

        #         # Displaying the description
        #         st.markdown("""
        #         **Description:**
        #         The git repository contains various TensorFlow projects and notebooks, each addressing different machine learning tasks. Highlights include:


        #         - **Callbacks_TensorFlow_MNIST:** Demonstrates using callbacks to improve MNIST digit classification.
        #         - **Convolution_NN_mnist:** Implements a convolutional neural network for MNIST classification.
        #         - **Happy_or_sad:** A model to classify images as happy or sad.
        #         - **Improve_MNIST_with_Convolutions:** Enhances MNIST classification using convolutional layers.
        #         - **Sign_Language_MNIST:** Classifies sign language digits using a neural network.
        #         - **Training_Validation_with_ImageDataGenerator:** Explores data augmentation techniques.
        #         - **Multiclass_Classifier:** Implements a multiclass classification model.
        #         """)

        #         # Displaying the tools used
        #         st.markdown("""
        #         **Tools Used:**

        #         **Python**, 
        #         **TensorFlow**, **Keras**, 
        #         **NumPy**, 
        #         **Matplotlib**,
        #         """)
        #         st.markdown(""" """)


        #         # Adding the GitHub link
        #         st.markdown("""**[GitHub](https://github.com/archanags001/tensorflow)**""")

        # with col1:
        #     with st.container(border=True):
        #         # Displaying the title of the project
        #         st.title("Recommendation System Using Pearson Correlation and Cosine Similarity")

        #         # Displaying the description
        #         st.markdown("""
        #         **Description:**
        #         This project implements a recommendation system using two different similarity metrics: Pearson Correlation and Cosine Similarity. The key tasks include:

        #         - **Data Preparation:** Loading and preprocessing the dataset.
        #         - **Pearson Correlation:** Calculating similarity between users/items using Pearson correlation.
        #         - **Cosine Similarity:** Calculating similarity between users/items using cosine similarity.
        #         - **Recommendation Generation:** Generating recommendations based on the computed similarities.
        #         """)

        #         # Displaying the tools used
        #         st.markdown("""
        #         **Tools Used:**

        #         **Python**,
        #         **Pandas**,  **NumPy**, 
        #         **Matplotlib**,
        #         """)
        #         st.markdown(""" """)


        #         # Adding the GitHub link
        #         st.markdown("""**[GitHub](https://github.com/archanags001/coding_challenge/blob/main/recommendation_Pearson_correlation_and_Cosine_similarity__.ipynb)**""")
        #         st.markdown(""" """)
        with col2:
            with st.container(border=True):
                # Displaying the title of the project
                st.title("Portfolio Explorer ")

                # Displaying the description
                st.markdown("""
                **Description:**
                The Portfolio Explorer is a Streamlit-based application designed to present a comprehensive and interactive personal portfolio. Key features include:

                - **Intro Page:** A dynamic introduction offering a professional overview.
                - **Resume Page:** A viewable and downloadable resume for quick access to detailed professional information.
                - **Experience Page:** An organized display of work experience, skills, and accomplishments.
                - **Projects Page:** A showcase of notable projects, including descriptions, technologies used, and links to repositories.
                
                 - **Contact Page:** An integrated contact form for easy communication and inquiries.
                """)

                # Displaying the tools used
                st.markdown("""
                **Tools Used:**

                **Python** , **Streamlit**
                """)
                st.markdown(""" """)

                c1, c2, c3, c4, c5 = st.columns(5)


                # Adding the GitHub link
                c1.markdown("""**[Link to app](https://portfolio2ra.streamlit.app/)**  """)
                c2.markdown("""**[GitHub](https://github.com/ravinder2023/portfolio1/blob/mainSSs)**""")

        # with col1:
        #     with st.container(border=True):
        #         st.markdown(""" """)
        #         st.markdown(""" """)
        #         # Displaying the title of the project
        #         st.title("Object Detection with YOLO")
        #         st.markdown(""" """)
        #         st.markdown(""" """)

        #         # Displaying the description
        #         st.markdown("""
        #         **Description:**
        #         The Object Detection project utilizes YOLO (You Only Look Once) to identify and classify objects within images efficiently. Key features include:

        #         - **Data Preparation:** Methods for preparing and preprocessing datasets tailored for YOLO object detection.
        #         - **Model Development:** Implementation of YOLO-based object detection models for real-time performance.
        #         - **Evaluation:** Techniques for assessing model accuracy and effectiveness, including visualizations of detection results.
        #         - **Application:** Demonstrations of applying the trained YOLO model to various images for accurate object detection.
        #         """)
        #         st.markdown(""" """)
        #         st.markdown(""" """)

        #         # Displaying the tools used
        #         st.markdown("""
        #         **Tools Used:**

        #         **Python**, **YOLO (You Only Look Once)** , **OpenCV**  , **Matplotlib**
        #         """)
        #         st.markdown(""" """)
        #         st.markdown(""" """)

        #         # Adding the GitHub link
        #         st.markdown("""**[GitHub](https://github.com/archanags001/ml_projects/blob/main/object_detection.pdf)**""")
        #         st.markdown(""" """)
        #         st.markdown(""" """)
        #         st.markdown(""" """)


        # with col2:
            # with st.container(border=True):
            #     # Displaying the title of the project
            #     st.title("Multimodal Biometric and Multi-Attack Protection Using Image Features")

            #     st.markdown("""
            #     **Description:** Multimodal biometrics is an integration of two or more biometric systems. It overcomes the limitations of other biometrics system like unimodal biometric system. Multimodal biometric for fake identity detection using image features uses three biometric patterns and they are iris, face, and fingerprint. In this system user chooses two biometric patterns as input, which will be fused. Gaussian filter is used to smooth this fused image. Smoothed version of input image and input image is compared using image quality assessment to extract image features. In this system different image quality measures are used for feature extraction. Extracted image features are used by artificial neural network to classify an image as real or fake. Depending on whether image is real or fake appropriate action is taken. Actions could be showing user identification on screen if image is classified as real or raising an alert if image is classified as fake. This system can be used in locker, ATM and other areas where personal identification is required.""")

            #     # Displaying the published paper link
            #     st.markdown("""
            #     **Published Paper:** [Multimodal Biometric and Multi-Attack Protection Using Image Features](http://pnrsolution.org/Datacenter/Vol3/Issue2/140.pdf)
            #     """)

