import streamlit as st
from streamlit_lottie import st_lottie
import json

def experience():
    col1,col2 =st.columns(2)

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
                <h2>Experience</h2>
            </div>
        """, unsafe_allow_html=True)
    path = "Animation_exp.json"
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
        col1,col2 = st.columns([3,2])
        col1.markdown(""" 
            ### AI & ML Industrial Training at National Institute of Electronics & Information Technology (NIELIT), Ropar 
            - Completed a six-month industrial training in AI and ML under the guidance of **Dr. Sarwan Singh**.
            - Gained hands-on experience with Python, AI, ML, Flask, Streamlit, and SQLite3.
            - Explored a variety of machine learning algorithms and computer vision applications.
            - Developed a computer vision project, implementing advanced image and video processing techniques.
            - Presented the computer vision project as a significant part of the training.
            - Enhanced problem-solving and project development skills by applying theoretical knowledge to real-world projects.
            - Gained technical expertise and practical experience in AI and ML, ready to tackle complex challenges in the field.

                            """)
        col2.markdown("""
            **Tools:**

            - Programming Languages: Python, SQL
            - Machine Learning: Scikit-Learn, TensorFlow, Keras
            - Data Visualization: Matplotlib, Seaborn.
            - Other Tools: Git, Colab Notebooks, Retool
            - Streamlit: Framework for creating interactive machine learning web apps.
            - computer vision, deep learning, NLP
            - SQLite3: Lightweight database for storing and managing data in machine learning applications.
            - OpenCV: Library for computer vision tasks (likely used in your computer vision project).
            - Scikit-learn: Machine learning library for implementing various ML algorithms.
            - TensorFlow/PyTorch (Optional, if used): Popular deep learning frameworks for building and training models.
            - Jupyter Notebooks: For interactive coding, analysis, and visualization of AI/ML models.
            """)
    with st.container():
        col1, col2 = st.columns([3, 2])
        col1.markdown("""
           ### Cloud Administration and Web Hosting Training at Guru Nanak Dev Engineering College 
    
              -  Duration: One month of hands-on training focused on cloud administration and web hosting.
              -  Developed a simple website using the Linux terminal.
              -  Gained experience in creating files and writing code directly in the Linux environment.
              -  Hosted the website using Linux-based tools and learned the process of deploying web applications.
              -  Wrote technical blogs and integrated them into the website.
              -  Used GitHub for version control, pushing code updates and managing repositories.
              -  Linked various web pages together to form a functional website.
              -  Acquired skills in Linux commands, version control, and website hosting as part of cloud administration.           """)
        col2.markdown(""" 
        **Tools:**
        
           - Linux Terminal: Command-line interface for interacting with the system and performing web hosting tasks.
           - GitHub: Version control platform for hosting and managing code repositories, as well as collaborating on projects.
           - Text Editors (e.g., Vim, Nano): Used for writing and editing code directly in the Linux terminal.
           - Apache/Nginx: Web servers used for hosting websites on Linux.
           - SSH: Secure method to remotely access and manage Linux servers for hosting purposes.
           - HTML/CSS/JavaScript: Core web technologies for creating and styling the website.
           - Git: Version control tool for pushing code to GitHub and managing project versions.
           - Bash Scripting: Used for automating tasks like file creation, deployment, and website management in Linux.  """)
    with st.container():
        col1, col2 = st.columns([3,2])

        col1.markdown("""
            ### Alert Enterprises Hackathon - Team Leader
            - Event: Hackathon organized by Alert Enterprises, a Sufi company, focused on innovative solutions for social good.
            - Role: Team Leader of my team.
            - Project: We developed a web application aimed at providing career guidance and support to meritorious students.
            - The web app was designed to assist students in navigating their academic and career paths by offering tools, resources, and personalized guidance.
            - Led the team through the entire development process, from brainstorming and planning to coding and deployment.
            - Worked collaboratively with my team to integrate features like career advice, scholarship opportunities, and educational resources to ensure the app met the needs of students.
            - Gained experience in team management, project development, and web app creation.

            """)
    #     col2.markdown("""
    #         **Tools:**
    
    #         - Programming Languages: Python
    #         - Machine Learning: TensorFlow, PyTorch
    #         - Image Processing: OpenCV, custom Python scripts
    #         - Object Detection: YOLO
    #         - Image Labeling:  LabelImg
    #         - Automation: Custom Python scripts
    #         """)
    # with st.container():
    #     col1, col2 = st.columns([3,2])
    #     col1.markdown("""
    #     ### Data Analyst –– [Centriqe Inc](https://bcentriqe.ai) (February 2020 - January 2021)
    #     - Developed an NLP system using NLTK to automate text analysis, resulting in a significant 39% reduction in manual analysis efforts, improving efficiency and accuracy.
    #     - Leveraged analytical and technical expertise to provide actionable insights and proposals, driving business improvement strategies.
    #     - Designed and implemented a range of predictive models, including classification and forecasting models, using various machine learning tools to solve complex business challenges.
    #     - Identified trends, key metrics, and critical data points, generating insightful dashboards using a variety of data visualization tools to facilitate data-driven decision-making.
    #         """)
    #     col2.markdown("""
    #     **Tools & Skills:**
    # - **Programming Languages:** Python, SQL
    # - **Libraries & Frameworks:** NLTK, scikit-learn, Pandas, NumPy
    # - **Machine Learning Tools:** Classification, Forecasting
    # - **Data Visualization Tools:** Tableau, Matplotlib, Seaborn, Power BI
    # - **Other:** Data Cleaning, Statistical Analysis, Predictive Modeling """)



    # # st.markdown()