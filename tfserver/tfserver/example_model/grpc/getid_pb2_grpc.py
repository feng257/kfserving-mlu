# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import getid_pb2 as getid__pb2


class InformationStub(object):
  """The information service definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.RequestID = channel.unary_unary(
        '/getid.Information/RequestID',
        request_serializer=getid__pb2.IDRequest.SerializeToString,
        response_deserializer=getid__pb2.IDReply.FromString,
        )


class InformationServicer(object):
  """The information service definition.
  """

  def RequestID(self, request, context):
    """Sends an ID
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_InformationServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'RequestID': grpc.unary_unary_rpc_method_handler(
          servicer.RequestID,
          request_deserializer=getid__pb2.IDRequest.FromString,
          response_serializer=getid__pb2.IDReply.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'getid.Information', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
