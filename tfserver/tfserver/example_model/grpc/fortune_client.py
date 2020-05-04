from __future__ import print_function
import grpc
import sys
import fortune_pb2
import fortune_pb2_grpc
import signal
import time

def sigint_handler(signum, frame):
    exit()


signal.signal(signal.SIGINT, sigint_handler)


def run(argv):
    channel = grpc.insecure_channel(str(argv[1]))
    stub = fortune_pb2_grpc.FortuneTellerStub(channel)
    response = stub.Predict(fortune_pb2.PredictionRequest())
    print(response.message)


if __name__ == '__main__':
    while True:
        # run(sys.argv)
        run(["", "127.0.0.1:50051"])
        time.sleep(5)
