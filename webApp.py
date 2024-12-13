from io import BytesIO
from showMap import locInfo
# import torch
import requests
import streamlit as st
from PIL import Image
import LandmarkClassification_Code
from showMap import show_map

# Define the Streamlit app
st.set_page_config(
    page_title="Google Landmark Detection",
    page_icon="https://www.logolynx.com/images/logolynx/16/169ba0bcc2e57b032548eeb606e4e7d5.png",
    layout="wide"
    )

cascade_style = """<style>

h2{
    text-align: center;
    font-family:math;
    color: white; 
    font-size: 30px;
    padding: 0;
    margin-bottom: 8px;
    font-weigth: bolder;
    letter-spacing: 1px;
    text-shadow: 2px 2px 2px black;
}

.stAlert.success {
    background-color: #D4EDDA;
    border: 1.5px;
    border-style: solid;
    border-color: black;
    color: Blue;
    padding: 2px;
    font-weight: bold;
    flex: 1;
    display: flex;
    justify-content: center;
    font-size: 24px;
    border-radius: 5px;
    font-family: cursive;
    letter-spacing: 1px;
    margin-bottom: 10px;
    box-shadow: 2.5px 3px 2.5px white
}

# .stAlert div{ 
#     background: inherit;
# }

# .stAlert p{
#     color: Blue;
#     font-family: cursive;
#     padding: 2px;
#     letter-spacing: 1px;
#     font-weight: bold;
#     font-size: 24px;
# }

.css-18ni7ap.e8zbici2{
    display: none;
}

.css-164nlkn.egzxvld1{
    display: none;
}

.main.css-uf99v8.egzxvld5{
    background: #7895806e;
}

.css-9s5bis.edgvbvh3{
    background: black;
    border-radius: 50%;
    box-shadow: 2.5px 3px 2.5px white
}

.e1fb0mya1.css-fblp2m.ex0cdmw0{
    color: white;
    font-weight: bolder;
}


.sidebar .sidebar-title{
    font-size: 24px;
    font-weight: bolder;
    color: green;
}

hr{
    margin: 0;
    padding: 0;
    border-width:2px;
    border-color:blue;
}

.h1_style{
    text-align: center;
    font-family:"Arial, Helvetica, Verdana, Georgia, Times New Roman, Open Sans, Roboto";
    color: Green; 
    font-size: 48px;
    padding: 0;
    margin-bottom: 8px;
    font-weigth: bolder;
    text-shadow: 2px 2px 2px white;
}

img{
    border: 2px;
    background: border-box;
    border-style: solid;
    border-color: black;
    border-radius: 6px;
    box-shadow: 2.5px 3px 2.5px white
}

iframe {
    border: 2px;
    width: 100%;
    border-style: solid;
    border-color: black;
    border-radius: 6px;
    box-shadow: 2.5px 3px 2.5px white
}

section{
    border: 2px;
    border-style: solid;
    border-radius: 5px;
    padding: 2.5px;
    margin: 5px;
    border-color: black;
    box-shadow: 2.5px 3px 2.5px white;
}

.stTextInput{
    border: 2px;
    border-style: solid;
    border-radius: 5px;
    padding: 6px;
    border-color: black;
    box-shadow: 2.5px 3px 2.5px white;
}

.stTextInput input{
    border: 2px;
    border-style: solid;
    border-radius: 5px;
    padding: 6px;
    border-color: black;
    box-shadow: 2.5px 3px 2.5px white;
}

label div p{
    font-style:italic;
    font-weight: bold;
    border: 2px;
    border-style: solid;
    border-radius: 5px;
    padding: 4px;
    border-color: black;
    box-shadow: 2.5px 3px 2.5px white;
}

.stRadio div{
    font-style: italic;
    font-family: cursive;
    font-weight: 500;
    margin-left: 15px;
}

.row-widget{
    border: 2px;
    border-style: solid;
    border-radius: 5px;
    padding: 2px;
    border-color: black;
    box-shadow: 2.5px 3px 2.5px white;
}

.css-ocqkz7 .e1tzin5v4{
    flex: 1;
    display: flex;
    justify-content: center;
}

.css-fis6aj.exg6vvm10{
    display: none;
}

.stAlert.info {
    background-color: #d1ecf1;
    color: #333333;
    border: 2px;
    border-style: solid;
    border-color: black;
    # font-weight: bold;
    border-radius: 5px;
    font-family: Roboto;
    padding: 5px;
    letter-spacing: 1px;
    margin-bottom: 10px;
    box-shadow: 2.5px 3px 2.5px white
}

</style>
"""

# Display a title with custom CSS style
st.markdown(cascade_style, unsafe_allow_html=True)
st.write("<h1 class='h1_style'>Google Landmark Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

def urlInput():
    url_input = st.text_input("Enter image URL here...")
    col1, col2 = st.columns(2)

    if url_input:
        name = ''
        response = requests.get(url_input, stream=True)
        # img = Image.open(response.raw)
        with col1:  
            img = Image.open(BytesIO(response.content))

            # st.image(img, caption='URL image', use_column_width=True)
            st.markdown(
                f'<img src="{url_input}" alt="example image" style="border: 3px solid green; margin-bottom:25px; width: 400px;">', 
                unsafe_allow_html=True
            )

            names = LandmarkClassification_Code.predict_landmarks(BytesIO(response.content), k=1)

            # Display the success message with custom CSS
            st.write("<div class='stAlert success'>{}</div>".format(names[0]), unsafe_allow_html=True)

            name = names[0]

        st.markdown("<hr>", unsafe_allow_html=True)
        destLat, destLon = show_map(name=name)

        with col2:
            patta = locInfo(destLat, destLon)
            st.write(f'''<div class='stAlert info'>
            <b>Name      :</b> {name} <br/>
            <b>Road   :</b> {patta[1]['road']} <br/>
            <b>Town      :</b> {patta[1]['town'] if "town" in patta[1] else (patta[1]['village'] if "village" in patta[1] else patta[1]['county'])} <br/>
            <b>Block     :</b> {patta[4]} <br/>
            <b>District  :</b> {patta[3]} <br/>
            <b>State     :</b> {patta[2]} <br/>
            <b>Country     :</b> {patta[1]['country']} <br/>
            <b>Country Code :</b> {patta[1]['country_code']} <br/>
            <b>ISO3166-2-lvl4 :</b> {patta[1]['ISO3166-2-lvl4']} <br/>
            <b>Post Code :</b> {patta[1]['postcode']} <br/>
            <b>Distance  :</b> {patta[0]} <br/>
            </div>''',unsafe_allow_html=True)



def imgInput():
    name = ''
    uploaded_file = st.file_uploader("Test Your Image Here...", type=['png', 'jpeg', 'jpg', 'JPG'])
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded image', use_column_width=True)

            names = LandmarkClassification_Code.predict_landmarks(uploaded_file, k=1)
            name = names[0]

            st.write("<div class='stAlert success'>{}</div>".format(name), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        destLat, destLon = show_map(name=name)

        with col2:
            patta = locInfo(destLat, destLon)
            st.write(f'''<div class='stAlert info'>
            <b>Name      :</b> {name} <br/>
            <b>Road   :</b> {patta[1]['road']} <br/>
            <b>Town      :</b> {patta[1]['town'] if "town" in patta[1] else (patta[1]['village'] if "village" in patta[1] else patta[1]['county'])} <br/>
            <b>Block     :</b> {patta[4]} <br/>
            <b>District  :</b> {patta[3]} <br/>
            <b>State     :</b> {patta[2]} <br/>
            <b>Country     :</b> {patta[1]['country']} <br/>
            <b>Country Code :</b> {patta[1]['country_code']} <br/>
            <b>ISO3166-2-lvl4 :</b> {patta[1]['ISO3166-2-lvl4']} <br/>
            <b>Post Code :</b> {patta[1]['postcode']} <br/>
            <b>Distance  :</b> {patta[0]} <br/>
            </div>''',unsafe_allow_html=True)


def main():
    # -- Sidebar
    st.sidebar.title('⚙️ Options')
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    datasrc = st.sidebar.radio("Select input source.", ['From Device', 'From URL'])
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)

    # # option = st.sidebar.radio("Select input type.", ['DEVICE', 'URL'])
    # if torch.cuda.is_available():
    #     deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    # else:
    #     deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # # -- End of Sidebar

    # st.header('Livestock Farming')
    # st.subheader('Select options left-haned menu bar.')
    # st.sidebar.markdown("https://github.com/thepbordin/Obstacle-Detection-for-Blind-people-Deployment")
    if datasrc == "From Device":    
        imgInput()
    elif datasrc == "From URL": 
        urlInput()

if __name__ == '__main__':
    main()
