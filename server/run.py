from flask import Flask, jsonify, request
from flask_cors import CORS

import aggregator
# import fake as aggregator

# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})


# sanity check route
@app.route('/hello', methods=['GET'])
def ping_pong():
    return jsonify('Hello World!!!')

@app.route('/coflows', methods=['GET'])
def getCoflows():
    data = aggregator.genCoflows()
    return jsonify(data)

@app.route('/traces', methods=['GET'])
def getTraces():
    data = aggregator.generateTraces()
    return jsonify(data)

@app.route('/mlfq', methods=['GET'])
def getMLFQ():
    qs = aggregator.genMLFQ()
    data = {}
    for i in range(len(qs)):
        data["q%s"%(i+1)] = [[i+1, e] for i, e in enumerate(qs[i])]
    # data["xdata"] = [str(i+1) for i in range(len(qs[0]))]
    # print(data)
    return jsonify(data)

@app.route('/cct', methods=['GET'])
def getCCT():
    data = aggregator.genCCT()
    return jsonify(data)

@app.route('/cct/transfer', methods=['GET','POST'])
def transfer():
    if request.method == 'POST':
        post_data = request.get_json()
        ending = post_data.get('ending')
        model = post_data.get('model')
        algo = post_data.get("algo")
        print("info:", ending, model)
        aggregator.transfer({
            "model": model,
        })
    aggregator.reset()
    return jsonify({"status": "200"})

@app.route('/cct/cancel', methods=['GET','POST'])
def cancel():
    if request.method == 'POST':
        post_data = request.get_json()
        ending = post_data.get('ending')
        model = post_data.get('model')
        print("cancel/info:", ending, model)
        aggregator.cancel()
    aggregator.reset()
    return jsonify({"status": "200"})

@app.route('/reset', methods=['GET','POST'])
def reset():
    if request.method == 'POST':
        post_data = request.get_json()
        post_data.get('info')
    aggregator.reset()
    return jsonify({"status": "200"})


@app.route('/traceSelect', methods=['GET','POST'])
def traceSelect():
    if request.method == 'POST':
        post_data = request.get_json()
        algo = post_data.get('algo')
        traceID = post_data.get('traceID')
        print("Trace选择：", algo, traceID)
        res = aggregator.configSource({
            "algo": algo,
            "traceID": traceID,
        })
        print("res:", res)
        if res:
            return jsonify({"status": "200"})
    return jsonify({"status": "400"})

@app.route('/system', methods=['GET'])
def getSystem():
    try:
        data = aggregator.generateSystem()
        # print(data)
        return jsonify(data)
    except TypeError as e:
        print(e)

@app.route('/getConfig', methods=['GET'])
def getConfig():
    data = aggregator.getConfig()
    return jsonify(data)

if __name__ == '__main__':
    try:
        app.run()
        print("Run over!")
    except KeyboardInterrupt:
        # monitor.stop()
        print("Receive KeyboardInterrupt!")