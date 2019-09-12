# **AI-Chat-Bot**
Implementation of a contextual AI chatbot for restaurant.

Created Simple deep learning chatbot using python 3.6, tensorflow and nltk. 
The chatbot is designed for a specific purpose like answering questions about a restaurant. 
Sample data to train the model is very less.
Used dictionary to handle the conversational context. 

Used Flask for hosting the website

**How to Run Application**

**Step 1**: Clone the repository
 ```https://github.com/srinivasshingade/AI-Chat-Bot.git```
 
 **Step 2**: Built the docker image
 ```docker build -t chatbotapp:latest .```
 
 **Step 3**: Run the cntainer
 ```docker run -d -p 5000:5000 chatbotapp```
 
 **Step 4** : Go to Web Browser
 Use default machine IP For ex, ```http://192.168.93.100:5000/``` to see the website.
 
 ![Website](https://user-images.githubusercontent.com/32945132/64753333-b1597d80-d4d7-11e9-8f7d-64acc921c009.png)
 
**Step 5**: Close the application
In your terminal, press Control+C to terminate the app.
