import streamlit as st
from streamlit_option_menu import option_menu
import base64
from streamlit_lottie import st_lottie
import requests
import json
from reume_page import resume
from experience_page import experience
#from upwork_page import feedbackRating
from project_page import projects
from contact_form import contact

 # Page setup
st.set_page_config(
    page_title="ravinder's portfolio",
    page_icon=".",
    layout="wide",
)


def gradient(color1, color2, color3, content1, content2):
    st.markdown(f'<h1 style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});font-size:60px;border-radius:2%;">'
                f'<span style="color:{color3};">{content1}</span><br>'
                f'<span style="color:white;font-size:17px;">{content2}</span></h1>',
                unsafe_allow_html=True)
def aboutMe():
    col1, col2 = st.columns(2)
    full_name = "Ravinder kaur"
    info = {'Intro': "Data scientist"}

    with col1:
        st.markdown("<h2 style='text-align: center; '>Hello! I'm Ravinder kaur 👋</h2>", unsafe_allow_html=True)

        st.markdown("""
        <style>
        .center-text {
        text-align: justify;
        }
        </style>
        <div class="justify-text">
        
        I am an enthusiastic and dedicated Computer Science Engineering student with a strong passion for web development, machine learning, and AI technologies. I have built a solid foundation in front-end and back-end development and have worked with technologies like HTML, CSS, JavaScript, and frameworks such as Flask, React, and Angular.

        Throughout my academic journey, I have developed a keen interest in applying AI to solve real-world problems. Some of the projects I’m currently working on include milk quality prediction using machine learning, text-to-image conversion, and an AI-powered legal documentation solution.

        I am always eager to learn new technologies and continuously seek opportunities to apply my skills in meaningful ways. I have actively participated in hackathons, internships, and held leadership roles, including my contributions to the Data Science Club and NCC Air Wing.

        With a passion for innovation and problem-solving, I am committed to making a significant impact in the tech industry. My ultimate goal is to become a skilled web developer while exploring emerging fields like AI, data science,
                </div>
        """, unsafe_allow_html=True)
        c1,c2, c3 =st.columns(3)
        c1.markdown("""**[GitHub](https://github.com/ravinder2023))**""")
        c2.markdown("""**[LinkedIn](https://www.linkedin.com/in/ravinder-kaur-658565270/)** """)
        c3.markdown("""**[Huggingface](https://huggingface.co/ravinder2024)**""")


    path = "Animation_blue_robo.json"
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
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get the base64 string of the image
logo_base64 = get_base64_image("RAAVI.jpeg")

# Logo styling
logo_html = f"""
    <style>
    .logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }}
    .logo {{
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
    }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}" class="logo">
    </div>
"""

# Display logo in the sidebar
st.sidebar.markdown(logo_html, unsafe_allow_html=True)
with st.sidebar:
    # Other sidebar elements
    # st.sidebar.image("logo_image.png", width=200, use_column_width=True)
    # Option menu in sidebar
    pages = ["About me", "Resume", "Experience",  "Projects", "Contact"]
    nav_tab_op = option_menu(
        menu_title="Ravinder",
        options=pages,
        icons=['person-fill', 'file-text', 'briefcase', 'folder', 'star', 'envelope'],
        menu_icon="cast",
        default_index=0,
    )
# Main content of the app
if nav_tab_op == "About me":
    aboutMe()

elif nav_tab_op == "Resume":
    resume()
elif nav_tab_op == "Experience":
    experience()
# elif nav_tab_op == "Testimonials":
#     feedbackRating()
elif nav_tab_op == "Projects":
    projects()
elif nav_tab_op == "Contact":
    contact()


