import requests
import streamlit as st
from streamlit_folium import folium_static
import folium
from geopy.geocoders import Nominatim
from geopy import distance

# Create a geolocator object
geolocator = Nominatim(user_agent='showMap')

def show_map(name):

    # Get the latitude and longitude coordinates of the location
    location = geolocator.geocode(name)
    latitude = location.latitude
    longitude = location.longitude

    print(latitude, longitude)
    print(location)

    # Create a map object
    m = folium.Map(location=[latitude, longitude], zoom_start=10)
    
    # Add a marker to the map
    folium.Marker(
        location=[latitude, longitude],
        popup=name,
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)

    # Add the map to the container
    st.write("<h2>Location of {}</h2>".format(name), unsafe_allow_html=True)
    folium_static(m)

    return latitude, longitude


def locInfo(lat, lon):
    curPlace = st.text_input("Enter your Source Location")

    # Get the latitude and longitude of the current location
    curLocPoint = geolocator.geocode(curPlace)
    curLatitude = curLocPoint.latitude
    curLongitude = curLocPoint.longitude

    curLocation = (curLatitude, curLongitude)
    destination = (lat, lon)

    distances = distance.distance(curLocation, destination).km

    # Get the state, district, and block of the location using the Nominatim API
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    response = requests.get(url).json()
    address = response["address"]
    block = address.get("city_district")
    district = address.get("state_district")
    state = address.get("state")

    return [distances, address, state, district, block]
