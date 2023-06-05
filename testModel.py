import requests

# Specify the file path
file_path = 'D:/fastAPI/111111111111112-8E36-4DA6-A898-B947CB9446AB-1436462707-1.1-m-26-dc.wav'

# Create a dictionary to hold the file object
files = {'file': open(file_path, 'rb')}

# Make a POST request to the API endpoint
url = 'http://127.0.0.1:5000/upload'
response = requests.post(url, files=files)

response_content = response.body

# Print the response content
print(response_content)