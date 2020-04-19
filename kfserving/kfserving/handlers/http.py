import json

import tornado.web


class HTTPHandler(tornado.web.RequestHandler):
    def initialize(self, models):
        self.models = models # pylint:disable=attribute-defined-outside-init

    def get_model(self, name):
        if name not in self.models:
            raise tornado.web.HTTPError(
                status_code=404,
                reason="Model with name %s does not exist." % name
            )
        model = self.models[name]
        if not model.ready:
            model.load()
        return model

    def validate(self, request):
        if "instances" not in request:
            raise tornado.web.HTTPError(
                status_code=400,
                reason="Expected key \"instances\" in request body"
            )

        if not isinstance(request["instances"], list):
            raise tornado.web.HTTPError(
                status_code=400,
                reason="Expected \"instances\" to be a list"
            )
        return request


class PredictHandler(HTTPHandler):
    def post(self, name):
        model = self.get_model(name)
        try:
            body = json.loads(self.request.body)
        except json.decoder.JSONDecodeError as e:
            raise tornado.web.HTTPError(
                status_code=400,
                reason="Unrecognized request format: %s" % e
            )
        request = model.preprocess(body)
        request = self.validate(request)
        response = model.predict(request)
        response = model.postprocess(response)
        self.write(response)


class ExplainHandler(HTTPHandler):
    def post(self, name):
        model = self.get_model(name)
        try:
            body = json.loads(self.request.body)
        except json.decoder.JSONDecodeError as e:
            raise tornado.web.HTTPError(
                status_code=400,
                reason="Unrecognized request format: %s" % e
            )
        request = model.preprocess(body)
        request = self.validate(request)
        response = model.explain(request)
        response = model.postprocess(response)
        self.write(response)
