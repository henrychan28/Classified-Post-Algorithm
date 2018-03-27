from flask import Flask
from flask import jsonify
from flask import request
from algorithms.hello import hello_world
import algorithms.backend_evaluation as BE
import algorithms.group_classification as GC



app = Flask(__name__)

@app.route('/', methods=['GET'])
def default():
	temp = GC.initialization()
	return jsonify(temp)

@app.route('/topic', methods=['POST'])
def topic():
	request_data = request.get_json()
	topic = request_data['topic']
	list_of_good_words = BE.good_words(topic)
	json_list = {'list_of_good_words': list_of_good_words}
	return jsonify(json_list)

@app.route('/title', methods=['POST'])
def title():
	request_data = request.get_json()
	title = request_data['title']
	topic = request_data['topic']
	score = {'score': BE.score(title, topic)}	
	return jsonify(score)


def startenv():
	GC.initialization()
	BE.initialization()

if __name__== "__main__":
	startenv()
	app.run()
