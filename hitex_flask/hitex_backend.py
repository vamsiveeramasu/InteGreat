from flask import Flask
from flask import request
from PyMongo import MongoClient

app = Flask(__name__)
mongo = MongoClient('localhost', 5000)
db = mongo['we<3mongo']

@app.route('/add', methods = ['POST'])
def add_to_workbook():
    if request.method == 'POST':
        jpeg = request.form['jpeg']
        name = request.form['name']

        if db.books.count() == 0 or db.books.name.count() == 0:
            db.books.name.insert_one({
                'name' : name,
                'images' : [jpeg]
            })

        else:
            db.books.update_one(
                {'name' : name},
                {"$addToSet" : {"images" : [jpeg] } }
            )


@app.route('/get', methods = ['GET'] )
def get_workbook():
    if request.method == 'GET':
        name = request.args.get['name']

        return db.name