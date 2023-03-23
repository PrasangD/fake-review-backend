# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask,session
from flask import Flask, redirect, url_for, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import testScrape as ts
import json as JSON
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
app.config.update(
    TEMPLATES_AUTO_RELOAD = True,
    TESTING=True,
    SECRET_KEY=b'_5#y2L"F4Q8z\n\xec]/'
)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/predict', methods=['GET', 'POST'])
# ‘/’ URL is bound with hello_world() function.
@cross_origin(supports_credentials=True)
def hello_world():
    f = request.data
    print(f)
    x = request.get_data('url').decode("UTF-8")
    print(x)
    result = ts.predict(x)
    session.clear()
    return result

@app.route('/refresh', methods=['GET','POST'])
@cross_origin(supports_credentials=True)
def refresh():
    session.clear()
    return JSON.dumps("Cleared")   

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run(debug=True)
