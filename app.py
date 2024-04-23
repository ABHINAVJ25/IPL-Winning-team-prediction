import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from streamlit_plotly_events import plotly_events

# Header
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown("<div style='text-align: center;'> Final Project for Semester Training </div>", unsafe_allow_html=True)
# st.markdown("<div style='text-align: center;'> on </div>", unsafe_allow_html=True)
# st.markdown("<div style='text-align: center;'> Estimating the chances of winning IPL using Machine Learning </div>", unsafe_allow_html=True)
# st.markdown("<div style='text-align: center;'> Submitted By : </div>", unsafe_allow_html=True)
# st.markdown("<div style='text-align: center;'>ABHINAV JAIN</div>", unsafe_allow_html=True)
# st.markdown("<div style='text-align: center;'>Roll No: 2003320</div>", unsafe_allow_html=True)


# st.markdown("<br><br>", unsafe_allow_html=True)
# st.markdown("<br><br>", unsafe_allow_html=True)




# Load your model
pipe = pickle.load(open('pipe.pkl', 'rb'))

from PIL import Image
bottom_image = st.file_uploader('ipl3', type='jpg', key=6)
if bottom_image is not None:
    image = Image.open(bottom_image)
    new_image = image.resize((800, 380))
    st.image(new_image)

st.title('IPL Win Predictor')
teams = sorted(['Sunrisers Hyderabad',
                'Mumbai Indians',
                'Royal Challengers Bangalore',
                'Kolkata Knight Riders',
                'Kings XI Punjab',
                'Chennai Super Kings',
                'Rajasthan Royals',
                'Delhi Capitals'])

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', teams)
with col2:
    bowling_team = st.selectbox('Select the bowling team', teams)

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

selected_city = st.selectbox('Cities', sorted(cities))

target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    wickets = st.number_input('Wickets', min_value=0, max_value=9)
with col5:
    overs = st.number_input('Overs completed', min_value=0, max_value=20)

# Initialize default values for r_2 and r_1
r_2 = 0
r_1 = 0

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - overs * 6
    wickets = 10 - wickets
    crr = score / overs
    rrr = runs_left * 6 / balls_left
    df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                       'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                       'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})
    result = pipe.predict_proba(df)
    r_1 = round(result[0][0] * 100)
    r_2 = round(result[0][1] * 100)
    st.header('Winning Probability')
    st.header(f"{batting_team}: {r_2}%")
    st.header(f"{bowling_team}: {r_1}%")

# Create a DataFrame for plotting
data = {'team': ['Batting Team', 'Bowling Team'],
        'probability': [r_2, r_1]}
df_plot = pd.DataFrame(data)

# Plotting
st.write("### Win Probability Comparison")
fig_bar = px.bar(df_plot, x='team', y='probability', color='team', labels={'team': 'Team', 'probability': 'Probability (%)'})

# Create a pie chart
fig_pie = px.pie(df_plot, values='probability', names='team', title='Win Probability Comparison')

# Display both the bar chart and the pie chart
st.plotly_chart(fig_bar)
st.plotly_chart(fig_pie)


# Footer
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown("<div style='text-align: center;'>ABHINAV JAIN</div>", unsafe_allow_html=True)
# st.markdown("<div style='text-align: center;'>Roll No: 2003320</div>", unsafe_allow_html=True)
