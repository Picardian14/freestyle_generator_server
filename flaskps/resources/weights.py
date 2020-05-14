from flask_restful import Resource

class Hip_Hop_weights(Resource):
    def get(self):
        return {'hello': 'weights'}

class Freestyle_weights(Resource):
    def get(self):
        return {'hello': 'freesrtyle weights'}