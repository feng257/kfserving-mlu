from __future__ import print_function
import grpc
import sys
import getid_pb2
import getid_pb2_grpc
import signal
import time

def sigint_handler(signum, frame):
    exit()


signal.signal(signal.SIGINT, sigint_handler)


def run(argv):
    channel = grpc.insecure_channel(str(argv[1]))
    stub = getid_pb2_grpc.InformationStub(channel)
    response = stub.RequestID(getid_pb2.IDRequest())
    print(response.message)


if __name__ == '__main__':
    while True:
        # run(sys.argv)
        run(["", "127.0.0.1:50051"])
        time.sleep(5)
