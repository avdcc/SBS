from flask import Flask,jsonify

app = Flask(__name__)

def hello():
    return "Hello World!"

@app.route("/")
def call():
  res = jsonify(result=hello())
  return res


@app.route("/test")
def testCall():
  res = jsonify(result=hello())
  return res

@app.route("/funcCall")
def testFunctionArgs():
  return "Call the function with argument by adding /argument to the end of the url"

@app.route("/funcCall/<arg>")

def func(arg):
  
  return "You used the argument: " + arg

@app.route("/sum/<int:add>/<int:add2>")
def add(add,add2):
  sum = add+add2
  return str(sum)
